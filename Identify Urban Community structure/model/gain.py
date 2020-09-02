import numpy as np
import main

def calculate_gain_professional_faster(node, co, cd, lamda0, lamda, mvcm_co_tmp, mvcm_cd_tmp, modularity_co_tmp,
                                       modularity_cd_tmp, last_cm, com_to_n, nc_co, nc_cd, c_num):
    # 计算modularity增益
    gain_modularity = modularity_cd_tmp[node] - modularity_co_tmp[node]
    if cd == -1:
        gain_modularity = gain_modularity + main.degree[node] / main.mm * (
                    np.sum(main.degree[com_to_n[co]]) - main.degree[node] - np.sum(main.degree[com_to_n[c_num]]))
    else:
        gain_modularity = gain_modularity + main.degree[node] / main.mm * (
                    np.sum(main.degree[com_to_n[co]]) - main.degree[node] - np.sum(main.degree[com_to_n[cd]]))
    modularity_co_tmp = modularity_co_tmp - main.adj[node]
    modularity_cd_tmp = modularity_cd_tmp + main.adj[node]
    # 计算MVCM增益
    node_feature = main.features[node]
    mvcm_co_tmp = mvcm_co_tmp - node_feature
    mvcm_cd_tmp = mvcm_cd_tmp + node_feature
    new_cm = (nc_co - 1) * CM_faster(mvcm_co_tmp, main.poi_num) + (nc_cd + 1) * CM_faster(mvcm_cd_tmp, main.poi_num)
    gain_mvcm = new_cm - last_cm

    gain_final = lamda0 * gain_modularity * main.node_num + m * lamda * gain_mvcm
    # gain_final = gain_modularity + 2*m*lamda*gain_mvcm

    return gain_final, mvcm_co_tmp, mvcm_cd_tmp, modularity_co_tmp, modularity_cd_tmp, new_cm


def CM_faster(p, poi_num):
    return 1 - np.sum(np.abs(p - 1) / poi_num)


def MVCM_faster(mvcm_np, poi_num, node_num, NC, c_num):
    re = 0
    for i in range(c_num): re += NC[i] * CM_faster(mvcm_np[i], poi_num)
    return re / node_num


def Modularity(com_to_nodes, adj, degree, m):
    modularity = 0
    for v in com_to_nodes.values():
        v_len = len(v)
        if v_len <= 1: continue
        for i in range(v_len - 1):
            for j in range(i + 1, v_len):
                modularity = modularity + adj[v[i], v[j]] - degree[v[i]] * degree[v[j]] / 2 / m
    return modularity / m