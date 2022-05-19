from model.calGain import CM_faster, MVCM_faster, Modularity

import numpy as np

def ha(c_num, partition_fast, adj, features, degree, mm, m, poi_num, node_num, lamda, threshold):
    # 初始化最大值追踪标记
    max_i = -1
    max_j = -1
    # 初始化社区adj和增益矩阵
    adj_evaluate = adj
    gain_total = np.zeros_like(adj)
    # 构建每个社区的点数向量
    NC = np.ones(c_num)
    # 维护目标优化功能dui
    mvcm_np = features / threshold
    # 初始化社区到节点列表的字典
    while 1:
        max_gain = -1

        if max_i == -1:
            for i in range(c_num - 1):
                for j in range(i + 1, c_num):
                    gain_tmp = 0
                    # 求modularity的增益
                    adj_tmp = adj_evaluate[i, j]
                    if adj_tmp != 0:
                        a = degree[partition_fast[i]]
                        b = degree[partition_fast[j]]
                        kk = np.sum(np.dot(a[:, np.newaxis], b[np.newaxis, :]))
                        gain = adj_tmp - kk / mm
                        gain_tmp += gain
                    else:
                        gain = 0
                    # 求mvcm的增益
                    gain_cm = (NC[i] + NC[j]) * CM_faster(mvcm_np[i] + mvcm_np[j], poi_num) - NC[i] * \
                            CM_faster(mvcm_np[i], poi_num) - NC[j] * CM_faster(mvcm_np[j], poi_num)
                    gain_tmp += lamda * m * gain_cm / node_num
                    # 聚合多目标增益
                    gain_total[i, j] = gain_tmp
                    if gain_tmp > max_gain:
                        max_gain = gain
                        max_i = i
                        max_j = j
        else:
            j = c_num - 1
            for i in range(c_num - 1):
                gain_tmp = 0
                # 求modularity的增益
                adj_tmp = adj_evaluate[i, j]
                if adj_tmp != 0:
                    a = degree[partition_fast[i]]
                    b = degree[partition_fast[j]]
                    kk = np.sum(np.dot(a[:, np.newaxis], b[np.newaxis, :]))
                    gain_tmp = adj_tmp - kk / mm
                # 求mvcm的增益
                gain_tmp += lamda * m * (
                            (NC[i] + NC[j]) * CM_faster(mvcm_np[i] + mvcm_np[j], poi_num) - NC[i] * CM_faster(mvcm_np[i],
                                                                                                              poi_num) - NC[
                                j] * CM_faster(mvcm_np[j], poi_num)) / node_num
                if gain_tmp > max_gain: max_gain = gain_tmp
                gain_total[i, j] = gain_tmp
            # 聚合增益
            max_i, max_j = np.unravel_index(gain_total.argmax(), gain_total.shape)
            if max_gain <= 0: max_gain = np.max(gain_total)

        if max_gain > 0:
            cm1 = partition_fast[max_i]
            cm2 = partition_fast[max_j]
            for i in range(max_i, max_j - 1): partition_fast[i] = partition_fast[i + 1]
            for i in range(max_j - 1, c_num - 2): partition_fast[i] = partition_fast[i + 2]
            partition_fast[c_num - 2] = cm1 + cm2
            del partition_fast[c_num - 1]

            c_num -= 1

            adj_evaluate = np.vstack((adj_evaluate, adj_evaluate[max_i, :] + adj_evaluate[max_j, :]))
            adj_evaluate = np.hstack(
                (adj_evaluate, (adj_evaluate[:, max_i] + adj_evaluate[:, max_j]).reshape(adj_evaluate.shape[0], 1)))
            adj_evaluate = np.delete(adj_evaluate, [max_i, max_j], axis=0)
            adj_evaluate = np.delete(adj_evaluate, [max_i, max_j], axis=1)
            adj_evaluate[-1, -1] = 0

            gain_total = np.vstack((gain_total, np.zeros_like(gain_total[0, :])))
            gain_total = np.hstack((gain_total, (np.zeros_like(gain_total[:, 0])).reshape(gain_total.shape[0], 1)))
            gain_total = np.delete(gain_total, [max_i, max_j], axis=0)
            gain_total = np.delete(gain_total, [max_i, max_j], axis=1)

            mvcm_np = np.vstack((mvcm_np, mvcm_np[max_i] + mvcm_np[max_j]))
            mvcm_np = np.delete(mvcm_np, [max_i, max_j], axis=0)

            NC = np.append(NC, NC[max_i] + NC[max_j])
            NC = np.delete(NC, [max_i, max_j])
        else:
            break

        # print('c_num is ', c_num, '; Modularity is ', Modularity(partition_fast, adj, degree, m), '; MVCM_NEW is ', MVCM_faster(mvcm_np, poi_num, node_num, NC, c_num))

    print('c_num is ', c_num, '; Modularity is ', Modularity(partition_fast, adj, degree, m),
          '; MVCM_NEW is ', MVCM_faster(mvcm_np, poi_num, node_num, NC, c_num))
    return partition_fast, c_num, mvcm_np, NC