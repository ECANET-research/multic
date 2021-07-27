import numpy as np
import networkx as nx
import random
import math
import time
import matplotlib.pyplot as plt
import pickle

random.seed(0)
np.random.seed(0)
# random.seed(8)
# np.random.seed(8)

np.set_printoptions(precision=4, suppress=True)

# Network statistics
N = 4000

# Degree distribution statistics
mu_in = 0.5
sigma_in = 1
mu_out = 0
sigma_out = math.sqrt(2)

# Generate degree sequences
samples = np.random.lognormal(mu_out, sigma_out, N)
out_deg_seq = [int(sample) for sample in samples]

out_deg_sum = sum(out_deg_seq)

samples = np.random.lognormal(mu_in, sigma_in, N)
in_deg_seq = [int(sample) for sample in samples]

in_deg_sum = sum(in_deg_seq)

print(len(out_deg_seq))
print(sum(out_deg_seq))
print(len(in_deg_seq))
print(sum(in_deg_seq))

for i in range(in_deg_sum - out_deg_sum):
	out_deg_seq[i] += 1

plt.hist(out_deg_seq, bins=np.logspace(np.log10(1), np.log10(max(out_deg_seq)), 50), log=True)
plt.gca().set_xscale('log')
plt.savefig('figures/out_deg_dist.png')

plt.hist(in_deg_seq, bins=np.logspace(np.log10(1), np.log10(max(in_deg_seq)), 50), log=True)
plt.gca().set_xscale('log')
plt.savefig('figures/in_deg_dist.png')

# Generate the network with the configuration model
G = nx.DiGraph(nx.directed_configuration_model(in_deg_seq, out_deg_seq, seed=0))
G.remove_edges_from(nx.selfloop_edges(G))
print('Number of nodes: %d' % G.number_of_nodes())
print('Number of edges: %d' % G.number_of_edges())

nx.write_edgelist(G, 'networks/network_%d_%.1f_%d_0.edgelist' % (N, mu_in, sigma_in))


random.seed(9)
np.random.seed(9)

np.set_printoptions(precision=4, suppress=True)

# Generate degree sequences
samples = np.random.lognormal(mu_out, sigma_out, N)
out_deg_seq = [int(sample) for sample in samples]

out_deg_sum = sum(out_deg_seq)

samples = np.random.lognormal(mu_in, sigma_in, N)
in_deg_seq = [int(sample) for sample in samples]

in_deg_sum = sum(in_deg_seq)

print(len(out_deg_seq))
print(sum(out_deg_seq))
print(len(in_deg_seq))
print(sum(in_deg_seq))

if out_deg_sum > in_deg_sum:
	for i in range(out_deg_sum - in_deg_sum):
		in_deg_seq[i] += 1
else:
	for i in range(in_deg_sum - out_deg_sum):
		out_deg_seq[i] += 1

# plt.hist(out_deg_seq, bins=np.logspace(np.log10(1), np.log10(max(out_deg_seq)), 50), log=True)
# plt.gca().set_xscale('log')
# plt.savefig('figures/out_deg_dist.png')

# plt.hist(in_deg_seq, bins=np.logspace(np.log10(1), np.log10(max(in_deg_seq)), 50), log=True)
# plt.gca().set_xscale('log')
# plt.savefig('figures/in_deg_dist.png')

# Generate the network with the configuration model
G = nx.DiGraph(nx.directed_configuration_model(in_deg_seq, out_deg_seq, seed=0))
G.remove_edges_from(nx.selfloop_edges(G))
print('Number of nodes: %d' % G.number_of_nodes())
print('Number of edges: %d' % G.number_of_edges())

nx.write_edgelist(G, 'networks/network_%d_%.1f_%d_1.edgelist' % (N, mu_in, sigma_in))
