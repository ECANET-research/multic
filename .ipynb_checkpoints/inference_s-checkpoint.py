import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.special import logit
from scipy.sparse import coo_matrix
import networkx as nx
import random
import math
import time
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import torch
import pickle
import EoN
from guppy import hpy

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

np.set_printoptions(precision=1, suppress=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

overall_start_time = time.time()
"""
Read the user network
"""
N = 1000
# N = 2000
# N = 4000
mu_in = 0.5
sigma_in = 1

K = 2
G_list = []
for k in range(K):
	G_list.append(nx.read_edgelist('networks/network_%d_%.1f_%d_%d.edgelist' % (N, mu_in, sigma_in, k), nodetype=int, create_using=nx.DiGraph))

alpha = np.zeros((N, N, K), dtype=np.float32)
for k in range(K):
	for e in G_list[k].edges:
		alpha[e[0], e[1], k] = random.uniform(0.01, 1)

alpha_agg = np.amax(alpha, axis=2)
E = np.count_nonzero(alpha_agg)

C_file = 500 * E
pi = np.zeros((C_file, K))
for c in range(C_file):
	pi[c, random.choice(range(K))] = 1

T = 10
t_threshold = 10

for ratio in [1, 2]:
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0)

	gamma = 4
	C = ratio * E

	max_iter = 500
	learning_rate = 0.5

	"""
	Parse simulated logs
	"""
	log_filename = 'logs/logs_%d_%d_%.1f.pkl' % (N, E, gamma)
	with open(log_filename, 'rb') as fp:
		sim_data = pickle.load(fp)

	data_nz = {}
	# Find nonzero cascades
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

	time_dict = {e:0 for e in range(N * N)}
	mask_dict = {c:set() for c in range(C)}
	time_diffs = []

	succ_edges = set()
	succ_users_cnt = []

	start_time = time.time()
	for c in range(C):
		if c % 500 == 0:
			print('Cascade %d' % c)

		sim_logs = data_nz[c]

		succ_users = []
		fail_users = set(range(N))

		for (t, u, j) in sim_logs:
			for (u, t_u) in succ_users:
				if u != j and t - t_u < t_threshold:
					idx = u * N + j
					mask_dict[c].add(idx)
					succ_edges.add(idx)
					time_dict[idx] += t - t_u
					time_diffs.append(t - t_u)
				else:
					break
			succ_users.insert(0, (j, t))
			fail_users.remove(j)

		for (j, t_j) in succ_users:
			for n in fail_users:
				idx = j * N + n
				time_dict[idx] += T - t_j

		succ_users_cnt.append(len(succ_users))

	end_time = time.time()
	parse_time = end_time - start_time

	start_time = time.time()
	succ_edges_list = list(succ_edges)
	succ_edges_dict = {k: v for v, k in enumerate(succ_edges_list)}
	n_succ_edges = len(succ_edges_list)
	print('Number of activated edges: %d' % n_succ_edges)

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

	print('Number of activation logs: %d' % succ_cnt)

	print('Data parsing time for %d nodes, %d edges, %d items: %.2fs' % (N, E, C, parse_time))
	print('Tensor construction time for %d nodes, %d edges, %d items: %.2fs' % (N, E, C, tensor_time))


	"""
	Conduct inference
	"""
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
	# opt = torch.optim.SGD([params_g], lr=learning_rate)
	opt = torch.optim.Adam([params_g], lr=learning_rate)
	# opt = torch.optim.LBFGS([params_g], lr=learning_rate, line_search_fn='strong_wolfe')

	infer_time = 0
	eval_time = 0

	alpha_agg = np.amax(alpha, axis=2)
	y_true = np.where(alpha_agg > 0, 1, 0).flatten()
	auc_list = []
	lik_list = []

	for i in range(max_iter):
		start_time = time.time()
		loss = objective(params_g)
		loss_val = loss.item()
		lik_list.append(loss_val)
		end_time = time.time()
		infer_time += end_time - start_time
		print('Iteration %d loss: %.4f' % (i, loss_val))
		if i > 100 and (lik_list[-2] - lik_list[-1]) / lik_list[-2] < 0.0001:
			break

		if i % 10 == 0:
			start_time = time.time()
			alpha_inferred_nonzero = torch.sigmoid(params_g).cpu().detach().numpy()
			alpha_inferred = np.zeros((N, N))
			for new_idx, old_idx in enumerate(succ_edges_list):
				i = old_idx // N
				j = old_idx % N
				alpha_inferred[i, j] = alpha_inferred_nonzero[new_idx]
			y_pred = alpha_inferred.flatten()

			auc = roc_auc_score(y_true, y_pred)
			print('AUC score: %.8f' % auc)
		# 	if len(auc_list) >= 1 and auc < auc_list[-1]:
		# 		break
			auc_list.append(auc)
			end_time = time.time()
			eval_time += end_time - start_time

		start_time = time.time()
		opt.zero_grad()
		loss.backward()
		opt.step()
		end_time = time.time()
		infer_time += end_time - start_time

	print('Optimization time for %d nodes, %d edges, %d items, gamma=%d: %ds for inference, %ds for evaluation' % (N, E, C, gamma, infer_time, eval_time))

	print('Cuda memory allocated: %.4f' % torch.cuda.memory_allocated())
	print('Max cuda memory allocated: %.4f' % torch.torch.cuda.max_memory_allocated())

	alpha_inferred_nonzero = torch.sigmoid(params_g).cpu().detach().numpy()

	alpha_inferred = np.zeros((N, N))
	for new_idx, old_idx in enumerate(succ_edges_list):
		i = old_idx // N
		j = old_idx % N
		alpha_inferred[i, j] = alpha_inferred_nonzero[new_idx]

	alpha_nonzero = alpha.flatten()[succ_edges_list]
	params_truth = torch.tensor(logit(alpha_nonzero), device=device)
	true_lik = objective(params_truth).item()

	print()
	print('Ground truth likelihood: %.4f' % true_lik)

	# print()
	# print('True alphas:')
	# print(alpha_agg)
	# print()
	# print('Inferred alphas:')
	# print(alpha_inferred)

	# acc_alpha = accuracy_score(np.where(alpha_inferred > 0.1, 1, 0).flatten(), np.where(alpha_agg > 0, 1, 0).flatten())

	# print()
	# print('Alpha accuracy: %.4f' % acc_alpha)
	# print()

	# print()
	# for threshold in np.linspace(0, 0.1, num=10, endpoint=False):
	# 	y_pred = np.where(alpha_inferred > threshold, 1, 0).flatten()
	# 	y_true = np.where(alpha_agg > 0, 1, 0).flatten()
	# 	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

	# 	# Print out threshold / percentage of recovered edges / percentage of false positive edges over all edges
	# 	print('%.3f / %.2f%% / %d / %d+%d / %d' % (threshold, tp * 100 / (tp + fn), (fn + tp), tp, fp, (tn + fp + fn + tp)))
	# print()

	res_dict = {}
	res_dict['succ_edges_list'] = succ_edges_list
	res_dict['alpha_inferred_nonzero'] = alpha_inferred_nonzero
	res_dict['auc'] = auc_list
	res_dict['lik'] = lik_list
	res_dict['true_lik'] = true_lik
	res_dict['time'] = {'parse': parse_time, 'tensor': tensor_time, 'infer': infer_time, 'eval': eval_time}
	res_dict['memory'] = torch.torch.cuda.max_memory_allocated()

	res_filename = 'res/s_%d_%d_%d_%.1f_lik.pkl' % (N, E, C, gamma)
	with open(res_filename, 'wb') as fp:
		pickle.dump(res_dict, fp)

overall_end_time = time.time()
print('Overall runtime: %.2fs' % (overall_end_time - overall_start_time))
