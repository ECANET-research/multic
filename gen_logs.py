import numpy as np
import networkx as nx
import random
import math
import time
import pickle
import EoN
import multiprocessing

random.seed(0)
np.random.seed(0)

print('Number of CPUs available: %d' % multiprocessing.cpu_count())

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
ratio = 110  # cascade-edge ratio that controls the number of cascades to generate

batch_size = multiprocessing.cpu_count()  # number of cascades generated in each batch


"""
Generate Ground Truth Parameter Values
"""
# Read network structure
G_list = []
for k in range(K):
	G_list.append(nx.read_edgelist('networks/network_%d_%.1f_%d_%d_%.2f.edgelist' % (N, mu_in, K, k, overlap), nodetype=int, create_using=nx.DiGraph))

# Sample edge transmission rates
alpha = np.zeros((N, N, K), dtype=np.float32)
for k in range(K):
	for e in G_list[k].edges:
		alpha[e[0], e[1], k] = random.uniform(0.01, 1)

# Set number of cascades to generate
alpha_agg = np.amax(alpha, axis=2)
E = np.count_nonzero(alpha_agg)
C = ratio * E

# Sample cascade layer membership values
pi = np.zeros((C, K))
for c in range(C):
	# Select the main layer the cascade spreads on
	layer = random.choice(range(K))

	# Add noise (layer mixing)
	noise = random.uniform(0, epsilon_max)
	pi[c, layer] = 1 - noise
	for l in range(K):
		if l != layer:
			pi[c, l] = noise / (K - 1)

# Save ground-truth parameter values to file
with open('truth/alpha_%d_%.1f_%d_%.1f_%.2f.pkl' % (N, mu_in, K, epsilon_max, overlap), 'wb') as fp:
	pickle.dump(alpha, fp)
with open('truth/pi_%d_%.1f_%d_%.1f_%.2f.pkl' % (N, mu_in, K, epsilon_max, overlap), 'wb') as fp:
	pickle.dump(pi, fp)


"""
Generate Spreading Logs by Simulating SIR Processes
"""
def simulate(c, sim_data):
	G_sim = nx.from_numpy_matrix(alpha_pi_prod[:, :, c % batch_size], create_using=nx.DiGraph)
	sim = EoN.fast_SIR(G_sim, 1, gamma, rho=1 / N, transmission_weight='weight', tmax=T, return_full_data=True)
	sim_data[c] = sim.transmissions()

# Generate cascades in batches, accelerated with multiprocessing
start_time = time.time()
manager = multiprocessing.Manager()
sim_data = manager.dict()

n_batches = (C - 1) // batch_size + 1
for b in range(n_batches):
	print('Batch %d/%d' % (b, n_batches))
	b_start_time = time.time()

	alpha_pi_prod = np.matmul(alpha, pi[b * batch_size:min(C, (b + 1) * batch_size), :].T)
	procs = []

	for c in range(b * batch_size, min(C, (b + 1) * batch_size)):
		queue = multiprocessing.Queue()
		queue.put(sim_data)
		p = multiprocessing.Process(target=simulate, args=(c, sim_data))  # each process simulates one cascade by executing the simulate function
		procs.append(p)
		p.start()
	for p in procs:
		p.join()

	b_end_time = time.time()
	print('Batch simulation time: %.2fs' % (b_end_time - b_start_time))

end_time = time.time()
print('Simulation time for %d nodes, %d edges, %d cascades: %.2fs' % (N, E, len(sim_data), (end_time - start_time)))

# Save spreading logs to file
log_filename = 'logs/logs_%d_%d_%d_%.1f_%.1f_%.2f.pkl' % (N, E, K, gamma, epsilon_max, overlap)
with open(log_filename, 'wb') as fp:
	pickle.dump(dict(sim_data), fp)

