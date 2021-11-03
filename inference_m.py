import numpy as np
from scipy.stats import spearmanr
import networkx as nx
import random
import time
import pickle
from itertools import permutations
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
overall_start_time = time.time()

"""
Set Parameters
"""
# Network parameters
N = 1000  # number of nodes
K = 2  # number of layers
overlap = 0  # layer overlap
mu_in = 0.5  # mu parameter of the in-degree sequence

# Spreading parameters
T = 10  # ending time
gamma = 2  # recovery rate
epsilon_max = 0  # maximum layer mixing
ratio = 16  # cascade-edge ratio

# Single layer result processing parameters
s_max_iter = 500  # maximum number of optimization iterations in the single layer phase
E_ratio = 1.1  # ratio of number of inferred edges over ground-truth edges

# Optimization parameters (multilayer phase)
s_c = 8  # cascade size threshold for filtering out small cascades
max_iter = 3000  # maximum number of optimization iterations
min_iter = 100  # minimum number of optimization iterations
learning_rate = 0.1  # initial learning rate of the Adam optimizer
tol = 1/1000000  # threshold of relative objective value change for stopping the optimization


"""
Read Ground-Truth Parameter Values
"""
G_list = []
for k in range(K):
	G_list.append(nx.read_edgelist('networks/network_%d_%.1f_%d_%d_%.2f.edgelist' % (N, mu_in, K, k, overlap), nodetype=int, create_using=nx.DiGraph))

with open('truth/alpha_%d_%.1f_%d_%.1f_%.2f.pkl' % (N, mu_in, K, epsilon_max, overlap), 'rb') as fp:
	alpha = pickle.load(fp)
with open('truth/pi_%d_%.1f_%d_%.1f_%.2f.pkl' % (N, mu_in, K, epsilon_max, overlap), 'rb') as fp:
	pi = pickle.load(fp)

alpha_agg = np.amax(alpha, axis=2)
E = np.count_nonzero(alpha_agg)
C_base = ratio * E


"""
Conduct Three Runs of Inference with Different Seeds
"""
for seed in [0, 1, 2]:
	with open('truth/alpha_%d_%.1f_%d_%.1f_%.2f.pkl' % (N, mu_in, K, epsilon_max, overlap), 'rb') as fp:
		alpha = pickle.load(fp)
	with open('truth/pi_%d_%.1f_%d_%.1f_%.2f.pkl' % (N, mu_in, K, epsilon_max, overlap), 'rb') as fp:
		pi = pickle.load(fp)

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# Read single layer inference results
	res_filename = 'results/s_%d_%d_%d_%d_%.1f_%.1f_%.2f_%d.pkl' % (N, E, C_base, K, gamma, epsilon_max, overlap, s_max_iter)
	with open(res_filename, 'rb') as fp:
		res_dict = pickle.load(fp)
	succ_edges_list = res_dict['succ_edges_list']
	alpha_inferred_nonzero = res_dict['alpha_inferred_nonzero']

	# Rank the edges by inferred edge weight in the single layer phase
	E_infer = int(E_ratio * E)
	alpha_inferred_sorted = np.partition(alpha_inferred_nonzero, len(alpha_inferred_nonzero) - E_infer)
	threshold = alpha_inferred_sorted[len(alpha_inferred_nonzero) - E_infer]  
		# edge weight threshold above which there are <E_infer> edges

	# Record the <E_infer> edges above the edge weight threshold
	edge_set = set()
	edge_list = []
	nonzero_edges = {}
	edge_cnt = 0
	for cnt, e in enumerate(succ_edges_list):
		if alpha_inferred_nonzero[cnt] >= threshold:
			i = e // N
			j = e % N
			edge_set.add(e)
			edge_list.append([i, j])
			nonzero_edges[e] = edge_cnt
			edge_cnt += 1

	# Read spreading logs
	log_filename = 'logs/logs_%d_%d_%d_%.1f_%.1f_%.2f.pkl' % (N, E, K, gamma, epsilon_max, overlap)
	with open(log_filename, 'rb') as fp:
		sim_data = pickle.load(fp)

	# Recognize cascades of size above the cascades size threshold
	data_nz = {}
	C_idx = []
	C_nz = 0
	C = 0
	for c in range(len(sim_data)):
		sim_logs = sim_data[c]
		if len(sim_logs) > 1:
			C_nz += 1
			if C_nz > C_base:
				break
			if len(sim_logs) > s_c:
				data_nz[C] = sim_logs
				C_idx.append(c)
				C += 1
	print('%d cascades out of %d above the cascade size threshold' % (C, C_base))
	pi_nz = pi[C_idx, :]

	# Construct data tensors from cascades
	mask_dict = {e: 0 for e in range(edge_cnt * C)}
	time_dict = {e: 0 for e in range(edge_cnt * C)}

	scatter_index_list = []

	start_time = time.time()
	for c in range(C):
		if c % 500 == 0 and c != 0:
			print('Finished parsing %d cascades' % c)
		sim_logs = data_nz[c]

		for e in edge_list:
			scatter_index_list.append(e[1] * C + c)

		succ_users = set()
		fail_users = set(range(N))

		for (t, u, j) in sim_logs:
			if j in fail_users:
				if t > T:
					break
				for (u, t_u) in succ_users:
					e = N * u + j
					if e in edge_set:
						idx = c * edge_cnt + nonzero_edges[e]
						mask_dict[idx] = 1
						time_dict[idx] += t - t_u
				succ_users.add((j, t))
				fail_users.remove(j)

		for (j, t_j) in succ_users:
			for n in fail_users:
				e = N * j + n
				if e in edge_set:
					idx = c * edge_cnt + nonzero_edges[e]
					time_dict[idx] += T - t_j

	end_time = time.time()
	parse_time = end_time - start_time

	start_time = time.time()
	mask = torch.tensor(list(mask_dict.values()), dtype=torch.uint8).view(C, edge_cnt).to(device)
	delta_t = torch.tensor(list(time_dict.values())).view(C, edge_cnt).to(device)
	scatter_index = torch.tensor(scatter_index_list, dtype=torch.int64).to(device)
	end_time = time.time()
	tensor_time = end_time - start_time

	print('Finished data parsing and tensor construction\n\nStarting optimization')


	"""
	Conduct Inference
	"""
	# Define objective function
	def objective(params):
		alpha_p = torch.sigmoid(params[:edge_cnt * K]).view(K, edge_cnt)
		pi_list = []
		for k in range(K - 1):
			pi_rm = 1
			for i in range(k):
				pi_rm -= pi_list[i]
			pi_list.append(torch.sigmoid(params[(edge_cnt * K + C * k):(edge_cnt * K + C * (k + 1))]) * pi_rm)
		pi_rm = 1
		for pi in pi_list:
			pi_rm -= pi
		pi_list.append(pi_rm)
		pi_p = torch.stack(pi_list, dim=-1).to(device)

		alpha_pi_prod = torch.matmul(pi_p, alpha_p)

		H = torch.zeros(N * C, dtype=torch.double).to(device).scatter_add_(0, scatter_index, torch.flatten(mask * alpha_pi_prod))
		H_nonzero = H[H != 0]

		return torch.sum(delta_t * alpha_pi_prod) - torch.sum(torch.log(H_nonzero))


	# Initialize parameters
	params_init = np.random.uniform(-5, 5, size=(edge_cnt * K + C * (K - 1)))
	params_g = torch.tensor(params_init, requires_grad=True, device=device)

	# Initialize optimizer
	opt = torch.optim.Adam([params_g], lr=learning_rate)

	infer_time = 0
	eval_time = 0

	lik_list = []
	pi_acc_list = []
	orders = list(permutations(range(K)))

	# Conduct optimization
	for i in range(max_iter):
		# Calculate objective
		start_time = time.time()
		loss = objective(params_g)
		loss_val = loss.item()
		lik_list.append(loss_val)
		end_time = time.time()
		infer_time += end_time - start_time

		print('Iteration %d loss: %.4f' % (i, loss_val))
		if len(lik_list) >= 2 and (lik_list[-2] - lik_list[-1]) / lik_list[-2] < tol:
			break

		# Parse inferred pi values
		start_time = time.time()
		alpha_inferred = torch.sigmoid(params_g[:edge_cnt * K]).view(K, edge_cnt).cpu().detach().numpy().T
		pi_list = []
		for k in range(K - 1):
			pi_rm = 1
			for i in range(k):
				pi_rm -= pi_list[i]
			pi_list.append(torch.sigmoid(params_g[(edge_cnt * K + C * k):(edge_cnt * K + C * (k + 1))]).cpu().detach() * pi_rm)
		pi_rm = 1
		for pi in pi_list:
			pi_rm -= pi
		pi_list.append(pi_rm)
		pi_inferred = torch.stack(pi_list, dim=-1)

		# Evaluate pi accuracy
		acc_pi_best = 0
		for order in orders:  # enumerate possible orders of layers
			acc_pi = accuracy_score(np.argmax(pi_inferred[:, order], axis=1), np.argmax(pi_nz, axis=1))
			if acc_pi > acc_pi_best:
				acc_pi_best = acc_pi
		print('Pi accuracy: %.4f' % acc_pi_best)
		pi_acc_list.append(acc_pi_best)
		end_time = time.time()
		eval_time += end_time - start_time

		# Loss propagation
		start_time = time.time()
		opt.zero_grad()
		loss.backward()
		opt.step()
		end_time = time.time()
		infer_time += end_time - start_time

	# Prepare for alpha accuracy evaluation:
	# for each layer, recognize cascades that spread on it and remove edges that are not activated by these cascades
	start_time = time.time()
	pi_b = torch.argmax(pi_inferred, dim=1).numpy()

	c_cluster = {k:set() for k in range(K)}
	for c in range(C):
		c_cluster[pi_b[c]].add(c)

	nz_edges_l = {k:set() for k in range(K)}
	for k in range(K):
		for c in c_cluster[k]:
			sim_logs = data_nz[c]
			sim_logs.sort()

			succ_users = set()
			fail_users = set(range(N))

			for (t, u, j) in sim_logs:
				if j in fail_users:
					if t > T:
						break
					for (u, t_u) in succ_users:
						e = N * u + j
						if e in edge_set:
							nz_edges_l[k].add(e)
					succ_users.add((j, t))
					fail_users.remove(j)

	alpha_inferred_nonzero = torch.sigmoid(params_g[:(edge_cnt * K)]).view(K, edge_cnt).cpu().detach().numpy().T
	alpha_inferred = np.zeros((N, N, K))
	for k in range(K):
		for e in nz_edges_l[k]:
			i = e // N
			j = e % N
			alpha_inferred[i, j, k] = alpha_inferred_nonzero[nonzero_edges[e], k]

	# Calculate metrics of alpha accuracy
	mae_best = 100
	ae_best = 100
	corr_best = -1
	roc_auc_best = 0
	prc_auc_best = 0
	order_best = None

	alpha_nz_idx = np.nonzero(alpha)
	a_true = alpha[alpha_nz_idx]
	y_true = np.where(alpha > 0, 1, 0).flatten()
	for order in orders:
		alpha_inferred_swap = alpha_inferred[:, :, order]
		a_pred = alpha_inferred_swap[alpha_nz_idx]
		ae = np.absolute(a_pred - a_true)
		mae = np.mean(ae / a_true)
		corr, _ = spearmanr(a_pred, a_true)
		roc_auc = roc_auc_score(y_true, alpha_inferred_swap.flatten())
		pr_all, re_all, _ = precision_recall_curve(y_true, alpha_inferred_swap.flatten())
		prc_auc = auc(re_all, pr_all)
		if prc_auc > prc_auc_best:
			mae_best = mae
			ae_best = np.mean(ae)
			corr_best = corr
			roc_auc_best = roc_auc
			prc_auc_best = prc_auc
			order_best = order

	print('Alpha MAE: %.4f' % ae_best)
	print('Alpha normalized MAE: %.4f' % mae_best)
	print('Alpha correlation: %.8f' % corr_best)
	print('Alpha AUC (ROC): %.8f' % roc_auc_best)
	print('Alpha AUC (PRC): %.8f' % prc_auc_best)
	print()

	end_time = time.time()
	eval_time += end_time - start_time

	print('Optimization time for N=%d, K=%d, overlap=%.2f, mu_in=%.1f, E=%d, C=%d, gamma=%d, epsilon_max=%.1f, s_c=%d: %ds for data parsing, %ds for tensor construction, %ds for inference, %ds for evaluation' % (N, K, overlap, mu_in, E, C, gamma, epsilon_max, s_c, parse_time, tensor_time, infer_time, eval_time))
	
	# Save results to file
	res_dict = {}
	res_dict['acc'] = {'pi_acc': pi_acc_list, 'alpha_corr': corr_best, 'alpha_mae': ae_best, 'alpha_nmae': mae_best, 'alpha_roc_auc': roc_auc_best, 'alpha_prc_auc': prc_auc_best}
	res_dict['lik'] = lik_list
	res_dict['time'] = {'parse': parse_time, 'tensor': tensor_time, 'infer': infer_time, 'eval': eval_time}
	if torch.cuda.is_available():
		res_dict['memory'] = torch.torch.cuda.max_memory_allocated()

	res_filename = 'results/m_%d_%d_%d_%d_%.1f_%d_%.1f_%.2f_%d_%d.pkl' % (N, E, K, C_base, gamma, s_c, epsilon_max, overlap, seed, s_max_iter)
	with open(res_filename, 'wb') as fp:
		pickle.dump(res_dict, fp)

overall_end_time = time.time()
print('Overall runtime: %.2fs' % (overall_end_time - overall_start_time))
