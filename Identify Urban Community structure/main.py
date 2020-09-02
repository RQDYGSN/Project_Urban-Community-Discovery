from initial import initial
from model.hierarchical_aggregation import ha
from model.node_level_adjustment import nla
from save_result import save_result

import numpy as np
import argparse

# Settings for Identifying Urban Community Structure
parser = argparse.ArgumentParser()
parser.add_argument('--lamda', type=float, default=0, help='A parameter of objective function.')
parser.add_argument('--swl', type=int, default=3, help='The sliding window length.')

args = parser.parse_args()

lamda = args.lamda
sliding_window_length = args.swl

# initial adj, features and threshold
adj, features, threshold = initial()

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
