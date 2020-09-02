from model.hierarchical_aggregation import ha
from model.node_level_adjustment import nla
from save_result import save_result

import numpy as np

# build ground truth for simple test
com_num = 3
node_per_com = 10
node_num = com_num * node_per_com

bias = 0
node_numpy = np.zeros(node_num)
for i in range(com_num):
    mu = np.random.rand() + bias
    bias += 0.8
    sigma = np.random.rand() / 6
    for j in range(node_per_com):
        node_numpy[i * node_per_com:(i + 1) * node_per_com:] = np.random.normal(mu, sigma, node_per_com)

adj = np.exp(-1 * np.square(node_numpy.reshape((node_num, 1)) - node_numpy.reshape((1, node_num))))
for i in range(node_num): adj[i, i] = 0

ground_truth = {}
for i in range(com_num): ground_truth[i] = []
for i in range(node_num):
    ground_truth[i//node_per_com].append(i)

A = node_per_com
threshold = np.array([A])
features = np.zeros((node_num, 1))
for i in range(node_num):
    features[i, 0] = A/node_per_com + (np.random.rand()-0.5)

# config parameters
lamda = 0.1
sliding_window_length = 3

# 周期数量
epoch_num = 100
# 读取节点数与特征类别数
node_num = adj.shape[0]
poi_num = features.shape[1]
# calculate m
mm = np.sum(adj)
m = mm/2
# 计算每个节点的度，用于模块度求解。
degree = np.sum(adj, axis=1)

# 读取初始的社区数量
c_num = node_num
partition_fast = {}
for i in range(c_num): partition_fast[i] = [i]

partition_fast, c_num, mvcm_np, NC = ha(c_num, partition_fast)
partition_fast = nla(c_num, partition_fast, mvcm_np, NC)
save_result(partition_fast)

