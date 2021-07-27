import numpy as np
import networkx as nx
import random
import math
import time
import matplotlib.pyplot as plt
import pickle
import EoN
import multiprocessing

random.seed(0)
np.random.seed(0)

np.set_printoptions(precision=4, suppress=True)

print(multiprocessing.cpu_count())


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


"""
Set true parameters
"""
T = 10

alpha = np.zeros((N, N, K), dtype=np.float32)
for k in range(K):
	for e in G_list[k].edges:
		alpha[e[0], e[1], k] = random.uniform(0.01, 1)

alpha_agg = np.amax(alpha, axis=2)
E = np.count_nonzero(alpha_agg)
print('%d edges' % E)
C = 1600 * E

pi = np.zeros((C, K))
for c in range(C):
	pi[c, random.choice(range(K))] = 1

G_sim = []
for k in range(K):
	G_sim.append(nx.from_numpy_matrix(alpha[:, :, k], create_using=nx.DiGraph))


"""
Generate activations using the Gillespie algorithm
"""
rho = 1 / N
gamma = 2

def simulate(c, sim_data):
	l = np.argmax(pi[c, :])

	sim = EoN.fast_SIR(G_sim[l], 1, gamma, rho=rho, transmission_weight='weight', tmax=T, return_full_data=True)

	sim_data[c] = sim.transmissions()

manager = multiprocessing.Manager()
sim_data = manager.dict()

start_time = time.time()

n_batches = (C - 1) // 1000 + 1
log_filename = 'logs/logs_%d_%d_%.1f.pkl' % (N, E, gamma)

for b in range(n_batches):
	print('Batch %d/%d' % (b, n_batches))
	b_start_time = time.time()
	procs = []
	for c in range(b * 1000, min(C, (b + 1) * 1000)):
		queue = multiprocessing.Queue()
		queue.put(sim_data)
		p = multiprocessing.Process(target=simulate, args=(c, sim_data))
		procs.append(p)
		p.start()

	for p in procs:
		p.join()

# 	if b % 100 == 99:
# 		with open(log_filename, 'wb') as fp:
# 			pickle.dump(dict(sim_data), fp)
	b_end_time = time.time()
	print('Batch simulation time: %.2fs' % (b_end_time - b_start_time))

end_time = time.time()
print(len(sim_data))
print('Simulation time for %d nodes, %d edges, %d items: %.2fs' % (N, E, C, (end_time - start_time)))

with open(log_filename, 'wb') as fp:
	pickle.dump(dict(sim_data), fp)

