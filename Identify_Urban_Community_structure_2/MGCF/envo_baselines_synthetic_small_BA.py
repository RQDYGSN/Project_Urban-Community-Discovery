# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:55:39 2020

@author: Liu Qinghe
"""

from karateclub import EgoNetSplitter, DANMF, NNSED, MNMF, BigClam, SymmNMF
from karateclub import GEMSEC, EdMot, SCD, LabelPropagation
from tkinter import _flatten
import pandas as pd 
import networkx as nx
from modularity_maximization import partition
import community
import random
import numpy as np
import math

epoch_num = 400

result_dict = {}
result_dict['EgoNetSplitter'] = []
result_dict['DANMF'] = []
result_dict['NNSED'] = []
result_dict['MNMF'] = []
result_dict['BigClam'] = []
result_dict['SymmNMF'] = []
result_dict['GEMSEC'] = []
result_dict['EdMot'] = []
result_dict['SCD'] = []
result_dict['LabelPropagation'] = []
result_dict['Louvain'] = []
# result_dict['Newman Spectral'] = []

c = ['Method', 'Min', '05%CI', 'Mean', '95%CI', 'Max']
result = []

clusters_number = 5

for epoch in range(epoch_num):
    # k = random.randint(1, 10)
    # clusters_number = k
    # average_cluster_size = node_num/k
    # shape = random.randint(1, 10)
    # G = nx.gaussian_random_partition_graph(n=node_num, s=average_cluster_size, v=shape, p_in=0.95, p_out=0.05, directed=False, seed=None)
    # G = G.to_undirected()
    N = random.randint(10, 100)
    G = nx.barabasi_albert_graph(N, int(math.log(N)))
    
    # G = nx.karate_club_graph()  # load a default graph

    # partition = community.best_partition(G)  # compute communities
    # pos = nx.spring_layout(G)  # compute graph layout
    # plt.figure(figsize=(8, 8))  # image is 8 x 8 inches
    # plt.axis('off')
    # nx.draw_networkx_nodes(G, pos, node_size=300, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
    # nx.draw_networkx_edges(G, pos, alpha=0.3)
    # print(evaluation.modularity(G, partition))
    # print(community.modularity(partition, G))
    # plt.show(G)
    
    # Overlapping
    model1 = EgoNetSplitter(weight='weight')
    model1.fit(G)
    cluster_membership1 = model1.get_memberships()
    clusters1 = cluster_membership1
    labels_pred1 = list(_flatten(list(clusters1.values())))
    result_dict['EgoNetSplitter'].append(community.modularity(dict(zip(list(range(len(labels_pred1))), labels_pred1)), G, weight='weight'))

    model2 = DANMF(layers=[32, clusters_number])
    # model2 = DANMF(layers=[32, 3])
    model2.fit(G)
    cluster_membership2 = model2.get_memberships()
    clusters2 = cluster_membership2
    labels_pred2 = list(_flatten(list(clusters2.values())))
    result_dict['DANMF'].append(community.modularity(dict(zip(list(range(len(labels_pred2))), labels_pred2)), G, weight='weight'))

    model3 = NNSED(dimensions=clusters_number)
    model3.fit(G)
    cluster_membership3 = model3.get_memberships()
    clusters3 = cluster_membership3
    labels_pred3 = list(_flatten(list(clusters3.values())))
    result_dict['NNSED'].append(community.modularity(dict(zip(list(range(len(labels_pred3))), labels_pred3)), G, weight='weight'))

    model4 = MNMF(clusters=clusters_number, iterations=200)
    model4.fit(G)
    cluster_membership4 = model4.get_memberships()
    clusters4 = cluster_membership4
    labels_pred4 = list(_flatten(list(clusters4.values())))
    result_dict['MNMF'].append(community.modularity(dict(zip(list(range(len(labels_pred4))), labels_pred4)), G, weight='weight'))

    model5 = BigClam(dimensions=clusters_number, iterations=50)
    model5.fit(G)
    cluster_membership5 = model5.get_memberships()
    clusters5 = cluster_membership5
    labels_pred5 = list(_flatten(list(clusters5.values())))
    result_dict['BigClam'].append(community.modularity(dict(zip(list(range(len(labels_pred5))), labels_pred5)), G, weight='weight'))

    model6 = SymmNMF(dimensions=clusters_number)
    model6.fit(G)
    cluster_membership6 = model6.get_memberships()
    clusters6 = cluster_membership6
    labels_pred6 = list(_flatten(list(clusters6.values())))
    result_dict['SymmNMF'].append(community.modularity(dict(zip(list(range(len(labels_pred6))), labels_pred6)), G, weight='weight'))

    # Non-overlapping
    model7 = GEMSEC(clusters=clusters_number)
    model7.fit(G)
    cluster_membership7 = model7.get_memberships()
    clusters7 = cluster_membership7
    labels_pred7 = list(_flatten(list(clusters7.values())))
    result_dict['GEMSEC'].append(community.modularity(dict(zip(list(range(len(labels_pred7))), labels_pred7)), G, weight='weight'))

    model8 = EdMot()
    model8.fit(G)
    cluster_membership8 = model8.get_memberships()
    clusters8 = cluster_membership8
    labels_pred8 = list(_flatten(list(clusters8.values())))
    result_dict['EdMot'].append(community.modularity(dict(zip(list(range(len(labels_pred8))), labels_pred8)), G, weight='weight'))

    model9 = SCD()
    model9.fit(G)
    cluster_membership9 = model9.get_memberships()
    clusters9 = cluster_membership9
    labels_pred9 = list(_flatten(list(clusters9.values())))
    result_dict['SCD'].append(community.modularity(dict(zip(list(range(len(labels_pred9))), labels_pred9)), G, weight='weight'))

    model10 = LabelPropagation()
    model10.fit(G)
    cluster_membership10 = model10.get_memberships()
    clusters10 = cluster_membership10
    labels_pred10 = list(_flatten(list(clusters10.values())))
    result_dict['LabelPropagation'].append(community.modularity(dict(zip(list(range(len(labels_pred10))), labels_pred10)), G, weight='weight'))

    labels_pred11 = community.best_partition(G)
    result_dict['Louvain'].append(community.modularity(labels_pred11, G))

    # labels_pred12 = partition(G)
    # result_dict['Newman Spectral'].append(community.modularity(labels_pred12, G))
    
    print(round((epoch+1)/epoch_num*100, 2), '% has been done.')
    
for k, v in result_dict.items():
    v_np = np.array(v)
    result.append([k, round(np.min(v_np), 3), round(np.quantile(v_np, 0.05), 3), round(np.mean(v_np), 3), round(np.quantile(v_np, 0.95), 3), round(np.max(v_np), 3)])
    
pd.DataFrame(result, columns=c).to_csv('./experiment_result/envo_baselines_synthetic_small_BA.csv')
print('done.')
