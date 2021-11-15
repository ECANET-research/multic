import numpy as np
import networkx as nx
import random
import math

"""
Set Parameters
"""
# Network parameters
N = 1000  # <W> mumber of nodes
K = 2  # <W> number of layers
overlap = 0  # <W> proportion of edge overlap between layers; only 0 and 1 are supported here

# Degree distribution parameters
mu_in = 0.5  # <W> mu parameter of the lognormal distribution for generating the in-degree sequence
sigma_in = 1  # sigma parameter of the lognormal distribution for generating the in-degree sequence
mu_out = 0  # mu parameter of the lognormal distribution for generating the out-degree sequence
sigma_out = math.sqrt(2 * (mu_in + (sigma_in ** 2) / 2 - mu_out))
	# sigma parameter of the lognormal distribution for generating the out-degree sequence
	# automatically calculated to match the expected means of in- and out-degree sequences

# Random seeds for generating each layer
seeds = [0, 9, 5, 6, 7]  # add more seeds if generating more than 5 layers


"""
Generate Network Structure
"""
for k in range(K):
	# Set seed
	if overlap == 0:
		seed = seeds[k]
	elif overlap == 1:
		seed = seeds[0]
	else:
		print('Error: layer overlap parameter set to a value not supported')
		exit()
	random.seed(seed)
	np.random.seed(seed)

	# Generate degree sequences
	samples = np.random.lognormal(mu_out, sigma_out, N)
	out_deg_seq = [int(sample) for sample in samples]

	samples = np.random.lognormal(mu_in, sigma_in, N)
	in_deg_seq = [int(sample) for sample in samples]

	# Make minor adjustments to exactly match total in-degree and out-degree
	out_deg_sum = sum(out_deg_seq)
	in_deg_sum = sum(in_deg_seq)

	if out_deg_sum > in_deg_sum:
		for i in range(out_deg_sum - in_deg_sum):
			in_deg_seq[i] += 1
	else:
		for i in range(in_deg_sum - out_deg_sum):
			out_deg_seq[i] += 1

	# Generate network with the directed configuration model, remove parallel edges and self-loops
	G = nx.DiGraph(nx.directed_configuration_model(in_deg_seq, out_deg_seq, seed=0))
	G.remove_edges_from(nx.selfloop_edges(G))

	print('Layer %d: %d nodes, %d edges' % (k + 1, G.number_of_nodes(), G.number_of_edges()))
	nx.write_edgelist(G, 'networks/network_%d_%.1f_%d_%d_%.2f.edgelist' % (N, mu_in, K, k, overlap))
