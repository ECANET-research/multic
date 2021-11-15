import numpy as np
from scipy.special import logit
from scipy.stats import spearmanr
import networkx as nx
import random
import time
import pickle
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score, precision_recall_curve, auc
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
overall_start_time = time.time()

"""
Set Paramters
"""
# Network parameters
N = 1000  # <RW> number of nodes
K = 2  # <RW> number of layers
overlap = 0  # <RW> layer overlap
mu_in = 0.5  # <RW> mu parameter of the in-degree sequence

# Spreading parameters
gamma = 2  # <RW> recovery rate in the SIR process
epsilon_max = 0  # <RW> maximum layer mixing
ratio = 16  # <W> cascade-edge ratio
T = 10  # ending time

# Optimization parameters
max_iter = 500  # <W> maximum number of optimization iterations
min_iter = 100  # minimum number of optimization iterations
learning_rate = 0.5  # initial learning rate of the Adam optimizer
tol = 0.0001  # threshold of relative objective value change for stopping the optimization


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
print('%d edges in the aggregated network' % E)
C = ratio * E


"""
Parse Spreading Logs
"""
log_filename = 'logs/logs_%d_%d_%d_%.1f_%.1f_%.2f.pkl' % (N, E, K, gamma, epsilon_max, overlap)
with open(log_filename, 'rb') as fp:
	sim_data = pickle.load(fp)

# Find nonzero cascades
data_nz = {}
C_nz_idx = []
C_nz = 0
C_real = len(sim_data)
for c in range(C_real):
	if len(sim_data[c]) > 1:
		data_nz[C_nz] = sim_data[c]
		C_nz_idx.append(c)
		C_nz += 1
print('%d nonzero cascades out of %d' % (C_nz, C_real))
pi_nz = pi[C_nz_idx, :]

# Construct data tensors from nonzero cascades
time_dict = {e:0 for e in range(N * N)}
mask_dict = {c:set() for c in range(C)}

succ_edges = set()

start_time = time.time()
for c in range(C):
	if c % 500 == 0 and c != 0:
		print('Finished parsing %d cascades' % c)

	sim_logs = data_nz[c]

	succ_users = []
	fail_users = set(range(N))

	for (t, u, j) in sim_logs:
		if t > T:
			break
		if j in fail_users:
			for (u, t_u) in succ_users:
				if u != j:
					idx = u * N + j
					mask_dict[c].add(idx)
					succ_edges.add(idx)
					time_dict[idx] += t - t_u
				else:
					break
			succ_users.insert(0, (j, t))
			fail_users.remove(j)

	for (j, t_j) in succ_users:
		for n in fail_users:
			idx = j * N + n
			time_dict[idx] += T - t_j

end_time = time.time()
parse_time = end_time - start_time

start_time = time.time()
succ_edges_list = list(succ_edges)
succ_edges_dict = {k: v for v, k in enumerate(succ_edges_list)}
n_succ_edges = len(succ_edges_list)

# Construct time tensor
time_val = np.array(list(time_dict.values()))[succ_edges_list]
delta_t = torch.FloatTensor(time_val).to(device)

# Construct mask and scatter tensor
mask_idx = []
scatter_idx = []
succ_cnt = 0
for c in range(C):
	mask_idx_list = []
	scatter_idx_list = []
	succ_cnt += len(mask_dict[c])
	for old_idx in mask_dict[c]:
		i = old_idx // N
		j = old_idx % N
		scatter_idx_list.append(j * C + c)
		mask_idx_list.append(succ_edges_dict[old_idx])
	mask_idx.append(torch.LongTensor(mask_idx_list).to(device))
	scatter_idx.append(torch.LongTensor(scatter_idx_list).to(device))
end_time = time.time()
tensor_time = end_time - start_time

print('Finished data parsing and tensor construction\n\nStarting optimization')


"""
Conduct Inference
"""
# Define objective function
def objective(params):
	alpha_p = torch.sigmoid(params)

	H = torch.zeros(N * C, device=device)
	for c in range(C):
		H.scatter_add_(0, scatter_idx[c], alpha_p.index_select(0, mask_idx[c]))

	H_nonzero = H[H != 0]

	return torch.sum(delta_t * alpha_p) - torch.sum(torch.log(H_nonzero))


# Initialize parameters
alpha_init = np.random.uniform(-5, 5, n_succ_edges)
params_g = torch.tensor(alpha_init, requires_grad=True, device=device, dtype=torch.float)

# Initialize optimizer
opt = torch.optim.Adam([params_g], lr=learning_rate)

infer_time = 0
eval_time = 0

y_true = np.where(alpha_agg > 0, 1, 0).flatten()
roc_auc_list = []
prc_auc_list = []
lik_list = []

# Conduct optimization
for it in range(max_iter):
	# Calculate objective
	start_time = time.time()
	loss = objective(params_g)
	loss_val = loss.item()
	lik_list.append(loss_val)
	end_time = time.time()
	infer_time += end_time - start_time

	print('Iteration %d loss: %.4f' % (it + 1, loss_val))

	# Evaluate accuracy
	start_time = time.time()
	alpha_inferred_nonzero = torch.sigmoid(params_g).cpu().detach().numpy()
	alpha_inferred = np.zeros((N, N))
	for new_idx, old_idx in enumerate(succ_edges_list):
		i = old_idx // N
		j = old_idx % N
		alpha_inferred[i, j] = alpha_inferred_nonzero[new_idx]
	y_pred = alpha_inferred.flatten()

	roc_auc = roc_auc_score(y_true, y_pred)
	pr, re, _ = precision_recall_curve(y_true, y_pred)
	prc_auc = auc(re, pr)
	print('ROC AUC score: %.8f' % roc_auc)
	print('PRC AUC score: %.8f' % prc_auc)
	roc_auc_list.append(roc_auc)
	prc_auc_list.append(prc_auc)
	end_time = time.time()
	eval_time += end_time - start_time

	# Stop optimization when relative decrease in objective value lower than threshold
	if it > min_iter and len(lik_list) >= 2 and (lik_list[-2] - lik_list[-1]) / lik_list[-2] < tol:
		break

	# Loss propagation
	start_time = time.time()
	opt.zero_grad()
	loss.backward()
	opt.step()
	end_time = time.time()
	infer_time += end_time - start_time

print('\nOptimization time for N=%d, K=%d, overlap=%.2f, mu_in=%.1f, E=%d, C=%d, gamma=%d, epsilon_max=%.1f: %ds for data parsing, %ds for tensor construction, %ds for inference, %ds for evaluation' % (N, K, overlap, mu_in, E, C, gamma, epsilon_max, parse_time, tensor_time, infer_time, eval_time))
if torch.cuda.is_available():
	print('Max cuda memory allocated: %.4f' % torch.torch.cuda.max_memory_allocated())

alpha_inferred_nonzero = torch.sigmoid(params_g).cpu().detach().numpy()

# Save results to file
res_dict = {}
res_dict['succ_edges_list'] = succ_edges_list
res_dict['alpha_inferred_nonzero'] = alpha_inferred_nonzero
res_dict['acc'] = {'roc_auc': roc_auc_list, 'prc_auc': prc_auc_list}
res_dict['lik'] = lik_list
res_dict['time'] = {'parse': parse_time, 'tensor': tensor_time, 'infer': infer_time, 'eval': eval_time}
if torch.cuda.is_available():
	res_dict['memory'] = torch.torch.cuda.max_memory_allocated()

res_filename = 'results/s_%d_%d_%d_%d_%.1f_%.1f_%.2f_%d.pkl' % (N, E, C, K, gamma, epsilon_max, overlap, max_iter)
with open(res_filename, 'wb') as fp:
	pickle.dump(res_dict, fp)

overall_end_time = time.time()
print('Overall runtime: %.2fs' % (overall_end_time - overall_start_time))
