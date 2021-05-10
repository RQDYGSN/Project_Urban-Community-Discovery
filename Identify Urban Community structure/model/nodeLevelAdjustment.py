from model.calGain import CM_faster, MVCM_faster, Modularity, calculate_gain_professional_faster

from tqdm import tqdm
import numpy as np
import random
import copy

def nla(c_num, partition_fast, mvcm_np, NC, sliding_window_length, node_num, adj, degree, mm, m, poi_num, epoch_num, lamda, features):
    # 创建滑动窗
    sliding_window = [1] * sliding_window_length
    sliding_window2 = [-1] * sliding_window_length

    # 维护目标优化功能堆
    # modularity_np
    modularity_np = np.zeros((c_num, node_num))
    for i in tqdm(range(c_num)):
        for j in range(node_num):
            modularity_np[i, j] = np.sum(adj[partition_fast[i], j])

    print('Begin to improve performance with combo ...')
    print('Epoch is 0; c_num is ', c_num, '; Modularity is ', Modularity(partition_fast, adj, degree, m), '; MVCM_NEW is ',
          MVCM_faster(mvcm_np, poi_num, node_num, NC, c_num))
    gain_update_threshold = 0.0000001
    # gain_update_threshold = 0
    for epoch in range(epoch_num):
        to_null_list = []
        # 更新滑动窗
        sliding_window = sliding_window[1::] + [0]
        sliding_window2 = sliding_window2[1::] + [0]
        co_list = list(range(c_num))
        cd_list = list(range(c_num)) + [-1]
        random.shuffle(co_list)
        random.shuffle(cd_list)

        for co in co_list:
            for cd in cd_list:
                if co in to_null_list: continue
                if cd in to_null_list: continue
                if co == cd: continue
                node_list = partition_fast[co]
                # 备份mvcm与modularity的dui
                mvcm_co_tmp = mvcm_np[co]
                modularity_co_tmp = modularity_np[co]
                nc_co_tmp = NC[co]
                if cd == -1:
                    mvcm_cd_tmp = np.zeros_like(mvcm_co_tmp)
                    modularity_cd_tmp = np.zeros_like(modularity_co_tmp)
                    random.shuffle(node_list)
                    nc_cd_tmp = 0
                else:
                    mvcm_cd_tmp = mvcm_np[cd]
                    modularity_cd_tmp = modularity_np[cd]
                    # 按照gain_list从大到小对node_list排序。
                    gain_list = modularity_cd_tmp[node_list]
                    ddd = dict(zip(node_list, gain_list))
                    ddd = sorted(ddd.items(), key=lambda x: x[1], reverse=True)
                    node_list = list(list(zip(*ddd))[0])
                    nc_cd_tmp = NC[cd]
                # 求解gain_acc来指导 聚合方向
                gain_acc_list = []
                acc_max = -1
                acc_max_i = -1
                point = 0
                # 保存初始的局部最佳dui
                mvcm_co_max = mvcm_co_tmp
                mvcm_cd_max = mvcm_cd_tmp
                modularity_co_max = modularity_co_tmp
                modularity_cd_max = modularity_cd_tmp
                nc_co_max = nc_co_tmp
                nc_cd_max = nc_cd_tmp
                # 计算初始cm
                last_cm = nc_co_tmp * CM_faster(mvcm_co_tmp, poi_num) + nc_cd_tmp * CM_faster(mvcm_cd_tmp, poi_num)
                # 备份partition_fast
                com_to_nodes_tmp = copy.deepcopy(partition_fast)
                if cd == -1: com_to_nodes_tmp[c_num] = []
                # 开始遍历节点
                for node in node_list:
                    # 计算局部增益
                    gain, mvcm_co_tmp, mvcm_cd_tmp, modularity_co_tmp, modularity_cd_tmp, last_cm = calculate_gain_professional_faster(
                        int(node), co, cd, 1, lamda, mvcm_co_tmp, mvcm_cd_tmp, modularity_co_tmp, modularity_cd_tmp,
                        last_cm, com_to_nodes_tmp, nc_co_tmp, nc_cd_tmp, c_num, degree, mm, m, adj, features, poi_num, node_num)
                    com_to_nodes_tmp[co].remove(node)
                    if cd == -1:
                        com_to_nodes_tmp[c_num].append(node)
                    else:
                        com_to_nodes_tmp[cd].append(node)
                    # 更新NC_tmp
                    nc_co_tmp = nc_co_tmp - 1
                    nc_cd_tmp = nc_cd_tmp + 1
                    # if 第一次保存增益值
                    if len(gain_acc_list) == 0:
                        gain_acc_list.append(0 + gain)
                        acc_max_i = point
                        acc_max = gain
                        mvcm_co_max = mvcm_co_tmp
                        mvcm_cd_max = mvcm_cd_tmp
                        modularity_co_max = modularity_co_tmp
                        modularity_cd_max = modularity_cd_tmp
                        nc_co_max = nc_co_tmp
                        nc_cd_max = nc_cd_tmp
                    # if 不是第一次保存增益值
                    else:
                        gain_acc_list.append(gain_acc_list[-1] + gain)
                        if gain_acc_list[-1] > acc_max:
                            acc_max_i = point
                            acc_max = gain_acc_list[-1]
                            mvcm_co_max = mvcm_co_tmp
                            mvcm_cd_max = mvcm_cd_tmp
                            modularity_co_max = modularity_co_tmp
                            modularity_cd_max = modularity_cd_tmp
                            nc_co_max = nc_co_tmp
                            nc_cd_max = nc_cd_tmp
                    # 更新当前节点的index of node_list
                    point = point + 1
                if acc_max > 0:
                    # 更新partition_fast以及dui
                    if cd == -1:
                        partition_fast[c_num] = node_list[:acc_max_i + 1:]
                        modularity_np = np.vstack((modularity_np, modularity_cd_max))
                        mvcm_np = np.vstack((mvcm_np, mvcm_cd_max))
                        NC = np.append(NC, nc_cd_max)
                        c_num += 1
                    else:
                        partition_fast[cd] = partition_fast[cd] + node_list[:acc_max_i + 1:]
                        modularity_np[cd] = modularity_cd_max
                        mvcm_np[cd] = mvcm_cd_max
                        NC[cd] = nc_cd_max
                    partition_fast[co] = node_list[acc_max_i + 1::]
                    modularity_np[co] = modularity_co_max
                    mvcm_np[co] = mvcm_co_max
                    NC[co] = nc_co_max
                    if len(partition_fast[co]) == 0: to_null_list.append(co)
                    # 更新滑动窗
                    sliding_window[-1] += acc_max
        # 处理to_null_list
        if len(to_null_list) > 0:
            to_alive_list = list(partition_fast.keys())
            NC = np.delete(NC, to_null_list)
            for i in to_null_list:
                to_alive_list.remove(i)
                c_num -= 1
            mvcm_np = mvcm_np[to_alive_list, :]
            modularity_np = modularity_np[to_alive_list, :]
            com_to_nodes_tmp = copy.deepcopy(partition_fast)
            partition_fast = {}
            for i in range(len(to_alive_list)): partition_fast[i] = com_to_nodes_tmp[to_alive_list[i]]
        sliding_window2[-1] = c_num
        if sum(sliding_window) <= gain_update_threshold and len(set(sliding_window2)) == 1: break
        # 显示一个epoch过去后的MVVCM以及modularity    MVCM_new(mvcm_np, threshold, poi_num, node_num, NC, c_num)
        print('Epoch is ', epoch + 1, '; c_num is ', c_num, '; Modularity is ', Modularity(partition_fast, adj, degree, m),
              '; MVCM_NEW is ', MVCM_faster(mvcm_np, poi_num, node_num, NC, c_num))

    return partition_fast