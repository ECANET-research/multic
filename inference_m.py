import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.special import logit
import networkx as nx
import random
import math
import time
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import torch
import pickle
import EoN
import sys

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
mu_in = 0
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


"""
Set true parameters
"""
T = 10
t_threshold = 10

# for gamma in [1, 2, 4, 8]:
# for ratio in [16]:
for rt_threshold in [8]:
	for seed in [0, 1, 2]:
		torch.cuda.empty_cache()
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)

# 		rt_threshold = 1
		gamma = 2
		ratio = 16

		C_base = ratio * E

		max_iter = 3000
		learning_rate = 0.1


		"""
		Read single-layer inference results
		"""
		res_filename = 'res/s_%d_%d_%d_%.1f_lik.pkl' % (N, E, C_base, gamma)
		with open(res_filename, 'rb') as fp:
			res_dict = pickle.load(fp)

		succ_edges_list = res_dict['succ_edges_list']
		alpha_inferred_nonzero = res_dict['alpha_inferred_nonzero']
		E_infer = int(1.1 * E)
		alpha_inferred_sorted = np.partition(alpha_inferred_nonzero, len(alpha_inferred_nonzero) - E_infer)
		threshold = alpha_inferred_sorted[len(alpha_inferred_nonzero) - E_infer]

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

		print(edge_cnt)

		# Recognize nonzero cascades
		log_filename = 'logs/logs_%d_%d_%.1f.pkl' % (N, E, gamma)
		with open(log_filename, 'rb') as fp:
			sim_data = pickle.load(fp)

		data_nz = {}
		C_idx = []
		C_nz = 0
		C = 0
		C_real = len(sim_data)
	# 	filter_threshold = 16
	# 	C_filter_idx = []
		for c in range(C_real):
			sim_logs = sim_data[c]
			if len(sim_logs) > 1:
				C_nz += 1
				if C_nz > C_base:
					break
				if len(sim_logs) > rt_threshold:
					data_nz[C] = sim_logs
					C_idx.append(c)
					C += 1
	# 			if len(sim_logs) > filter_threshold:
	# 				C_filter_idx.append(C_nz)
		print('%d cascades out of %d above the RT threshold' % (C, C_file))
		pi_nz = pi[C_idx, :]


		"""
		Parse simulated logs
		"""
		mask_dict = {e: 0 for e in range(edge_cnt * C)}
		time_dict = {e: 0 for e in range(edge_cnt * C)}

		scatter_index_list = []

		start_time = time.time()
		for c in range(C):
			if c % 500 == 0:
				print('Cascade %d' % c)
			sim_logs = data_nz[c]

			for e in edge_list:
				scatter_index_list.append(e[1] * C + c)

			succ_users = set()
			fail_users = set(range(N))

			for (t, u, j) in sim_logs:
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
		# H_index = torch.tensor(H_index_list).to(device)
		scatter_index = torch.tensor(scatter_index_list, dtype=torch.int64).to(device)
		end_time = time.time()
		tensor_time = end_time - start_time

		print('Data parsing time for %d nodes, %d edges, %d items: %.2fs' % (N, E, C, parse_time))
		print('Tensor construction time for %d nodes, %d edges, %d items: %.2fs' % (N, E, C, tensor_time))


		"""
		Conduct inference
		"""
		def objective(params):
			alpha_p = torch.sigmoid(params[:edge_cnt * K]).view(K, edge_cnt)
			pi_sl = torch.sigmoid(params[edge_cnt * K:])
			pi_p = torch.stack((pi_sl, 1 - pi_sl), dim=-1)

			alpha_pi_prod = torch.matmul(pi_p, alpha_p)

			H = torch.zeros(N * C, dtype=torch.double).to(device).scatter_add_(0, scatter_index, torch.flatten(mask * alpha_pi_prod))
			H_nonzero = H[H != 0]

			return torch.sum(delta_t * alpha_pi_prod) - torch.sum(torch.log(H_nonzero))


		# Initialize parameters
		params_init = np.random.uniform(-5, 5, size=(edge_cnt * K + C))
		params_g = torch.tensor(params_init, requires_grad=True, device=device)

		# opt = torch.optim.SGD([params_g], lr=learning_rate)
		opt = torch.optim.Adam([params_g], lr=learning_rate)

		infer_time = 0
		eval_time = 0

		alpha_nonzero = np.zeros((edge_cnt, K))
		for e in edge_list:
			alpha_nonzero[nonzero_edges[e[0] * N + e[1]], :] = alpha[e[0], e[1], :]

		lik_list = []
		alpha_corr_list = []
		alpha_mae_list = []
		pi_corr_list = []
		pi_mae_list = []
		pi_acc_list = []
		for i in range(max_iter):
			start_time = time.time()
			loss = objective(params_g)
			loss_val = loss.item()
			lik_list.append(loss_val)
			print('Iteration %d loss: %.4f' % (i, loss_val))
			if len(lik_list) >= 2 and (lik_list[-2] - lik_list[-1]) / lik_list[-2] < 1/1000000:
				break
			end_time = time.time()
			infer_time += end_time - start_time

		# 	if i % 20 == 0:
			start_time = time.time()
			alpha_inferred = torch.sigmoid(params_g[:edge_cnt * K]).view(K, edge_cnt).cpu().detach().numpy().T
			pi_inferred_sl = torch.sigmoid(params_g[edge_cnt * K:]).cpu().detach().numpy()
			pi_inferred = np.stack((pi_inferred_sl, 1 - pi_inferred_sl), axis=-1)
			pi_inferred_b = np.where(pi_inferred > 0.5, 1, 0)

			alpha_inferred_swap = alpha_inferred[:, [1, 0]]
			r1_nz_alpha, p1_nz_alpha = spearmanr(alpha_inferred, alpha_nonzero, axis=None)
			r2_nz_alpha, p2_nz_alpha = spearmanr(alpha_inferred_swap, alpha_nonzero, axis=None)
			mae1_nz_alpha = mean_absolute_error(alpha_inferred.flatten(), alpha_nonzero.flatten())
			mae2_nz_alpha = mean_absolute_error(alpha_inferred_swap.flatten(), alpha_nonzero.flatten())

			r1_pi, p1_pi = spearmanr(pi_inferred_b[:, 0], pi_nz[:, 0], axis=None)
			r2_pi, p2_pi = spearmanr(pi_inferred_b[:, 1], pi_nz[:, 0], axis=None)
			mae1_pi = mean_absolute_error(pi_inferred[:, 0], pi_nz[:, 0])
			mae2_pi = mean_absolute_error(pi_inferred[:, 1], pi_nz[:, 0])
			acc1_pi = accuracy_score(np.argmax(pi_inferred_b, axis=1), np.argmax(pi_nz, axis=1))
			acc2_pi = accuracy_score(np.argmax(pi_inferred_b[:, [1, 0]], axis=1), np.argmax(pi_nz, axis=1))

		# 	acc1_pi_filter = accuracy_score(np.argmax(pi_inferred_b[C_filter_idx, :], axis=1), np.argmax(pi_nz[C_filter_idx, :], axis=1))
		# 	pi_inferred_b_rv = pi_inferred_b[:, [1, 0]]
		# 	acc2_pi_filter = accuracy_score(np.argmax(pi_inferred_b_rv[C_filter_idx, :], axis=1), np.argmax(pi_nz[C_filter_idx, :], axis=1))

			print('Non-zero alpha correlation: %.4f' % max(r1_nz_alpha, r2_nz_alpha))
			alpha_corr_list.append(max(r1_nz_alpha, r2_nz_alpha))
			print('Pi correlation: %.4f' % max(r1_pi, r2_pi))
			pi_corr_list.append(max(r1_pi, r2_pi))
			print('Non-zero alpha MAE: %.4f' % max(mae1_nz_alpha, mae2_nz_alpha))
			alpha_mae_list.append(max(mae1_nz_alpha, mae2_nz_alpha))
			print('Pi MAE: %.4f' % min(mae1_pi, mae2_pi))
			pi_mae_list.append(min(mae1_pi, mae2_pi))
			print('Pi accuracy: %.4f' % max(acc1_pi, acc2_pi))
		# 	print('Filtered pi accuracy: %.4f' % max(acc1_pi_filter, acc2_pi_filter))
			pi_acc_list.append(max(acc1_pi, acc2_pi))
			end_time = time.time()
			eval_time += end_time - start_time
		# 	if len(pi_acc_list) >= 2 and pi_acc_list[-1] < pi_acc_list[-2]:
		# 		break

			start_time = time.time()
			opt.zero_grad()
			loss.backward()
			opt.step()
		# 	torch.cuda.empty_cache()
			end_time = time.time()
			infer_time += end_time - start_time

		print('Cuda memory allocated: %.4f' % torch.cuda.memory_allocated())
		print('Max cuda memory allocated: %.4f' % torch.torch.cuda.max_memory_allocated())

		params_truth = torch.tensor(np.concatenate([logit(alpha_nonzero.T).flatten(), logit(pi_nz[:, 0]).flatten()]), device=device)
		true_lik = objective(params_truth).item()

		print()
		print('Ground truth likelihood: %.4f' % true_lik)
		print('Likelihood error ratio: %.5f' % (abs(true_lik - loss_val) / true_lik))
		print()

		print('Optimization time for %d nodes, %d edges, %d items, gamma=%.1f: %ds for inference, %ds for evaluation' % (N, E, C, gamma, infer_time, eval_time))

		res_dict = {}
		res_dict['acc'] = {'alpha_corr': alpha_corr_list, 'alpha_mae': alpha_mae_list, 'pi_corr': pi_corr_list, 'pi_mae': pi_mae_list, 'pi_acc': pi_acc_list}
		res_dict['lik'] = lik_list
		res_dict['true_lik'] = true_lik
		res_dict['time'] = {'parse': parse_time, 'tensor': tensor_time, 'infer': infer_time, 'eval': eval_time}
		res_dict['memory'] = torch.torch.cuda.max_memory_allocated()

		res_filename = 'res/m_%d_%d_%d_%.1f_%d_%d_lik.pkl' % (N, E, C_base, gamma, rt_threshold, seed)
		with open(res_filename, 'wb') as fp:
			pickle.dump(res_dict, fp)

overall_end_time = time.time()
print('Overall runtime: %.2fs' % (overall_end_time - overall_start_time))
