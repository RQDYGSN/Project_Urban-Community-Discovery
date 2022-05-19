# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:19:08 2020

@author: Liu Qinghe
"""
import networkx as nx
from karateclub import EgoNetSplitter, DANMF, NNSED, MNMF, BigClam, SymmNMF
from karateclub import GEMSEC, EdMot, SCD, LabelPropagation
from tkinter import _flatten
import geopandas as gpd
import pandas as pd 
from tqdm import tqdm
import networkx as nx
import community

# 确认节点数量
bs_gpd = gpd.read_file('../../data/basestation-gis/basestation_aggregation.shp')
bs_gpd = bs_gpd.iloc[::, ::]
bs_num = bs_gpd.shape[0]

# 构造图节点
nodes_list = list(range(bs_num))
G = nx.Graph()
G.add_nodes_from(nodes_list)

# 读取邻接矩阵adjacent
# 以工作日(周一)的早高峰时段为例
adj = pd.read_csv('../../data/adj/adj_20151109_0002.csv')
adj = adj.iloc[:, 1:].values
adj = adj[::, ::]
adj = adj + adj.T
for i in range(bs_num): 
    adj[i, i] = 0

for i in range(bs_num):
    if (adj[i]==0).all():
        adj[i, :] = 0.0001
        adj[:, i] = 0.0001
        
for i in tqdm(range(bs_num)):
    for j in range(bs_num):
        tmp = adj[i, j]
        if tmp>0: G.add_edge(i, j, weight=tmp)
        
result = []
c = ['Method', 'Cluster_num', 'Modularity']

clusters_number = 20

# Overlapping
model1 = EgoNetSplitter(weight='weight')
model1.fit(G)
cluster_membership1 = model1.get_memberships()
clusters1 = cluster_membership1
labels_pred1 = list(_flatten(list(clusters1.values())))
modularity = community.modularity(dict(zip(list(range(len(labels_pred1))), labels_pred1)), G, weight='weight')
result.append(['EgoNetSplitter', len(set(labels_pred1)), modularity])
pd.DataFrame(result, columns=c).to_csv('./experiment_result/envo_baselines_real_full.csv')
print(1)

model2 = DANMF(layers=[32, clusters_number])
# model2 = DANMF(layers=[32, 3])
model2.fit(G)
cluster_membership2 = model2.get_memberships()
clusters2 = cluster_membership2
labels_pred2 = list(_flatten(list(clusters2.values())))
modularity = community.modularity(dict(zip(list(range(len(labels_pred2))), labels_pred2)), G, weight='weight')
result.append(['DANMF', len(set(labels_pred2)), modularity])
pd.DataFrame(result, columns=c).to_csv('./experiment_result/envo_baselines_real_full.csv')
print(2)

model3 = NNSED(dimensions=clusters_number)
model3.fit(G)
cluster_membership3 = model3.get_memberships()
clusters3 = cluster_membership3
labels_pred3 = list(_flatten(list(clusters3.values())))
modularity = community.modularity(dict(zip(list(range(len(labels_pred3))), labels_pred3)), G, weight='weight')
result.append(['NNSED', len(set(labels_pred3)), modularity])
pd.DataFrame(result, columns=c).to_csv('./experiment_result/envo_baselines_real_full.csv')
print(3)

model4 = MNMF(clusters=clusters_number, iterations=200)
model4.fit(G)
cluster_membership4 = model4.get_memberships()
clusters4 = cluster_membership4
labels_pred4 = list(_flatten(list(clusters4.values())))
modularity = community.modularity(dict(zip(list(range(len(labels_pred4))), labels_pred4)), G, weight='weight')
result.append(['MNMF', len(set(labels_pred4)), modularity])
pd.DataFrame(result, columns=c).to_csv('./experiment_result/envo_baselines_real_full.csv')
print(4)

model5 = BigClam(dimensions=clusters_number, iterations=50)
model5.fit(G)
cluster_membership5 = model5.get_memberships()
clusters5 = cluster_membership5
labels_pred5 = list(_flatten(list(clusters5.values())))
modularity = community.modularity(dict(zip(list(range(len(labels_pred5))), labels_pred5)), G, weight='weight')
result.append(['BigClam', len(set(labels_pred5)), modularity])
pd.DataFrame(result, columns=c).to_csv('./experiment_result/envo_baselines_real_full.csv')
print(5)

model6 = SymmNMF(dimensions=clusters_number)
model6.fit(G)
cluster_membership6 = model6.get_memberships()
clusters6 = cluster_membership6
labels_pred6 = list(_flatten(list(clusters6.values())))
modularity = community.modularity(dict(zip(list(range(len(labels_pred6))), labels_pred6)), G, weight='weight')
result.append(['SymmNMF', len(set(labels_pred6)), modularity])
pd.DataFrame(result, columns=c).to_csv('./experiment_result/envo_baselines_real_full.csv')
print(6)

# Non-overlapping
model7 = GEMSEC(clusters=clusters_number)
model7.fit(G)
cluster_membership7 = model7.get_memberships()
clusters7 = cluster_membership7
labels_pred7 = list(_flatten(list(clusters7.values())))
modularity = community.modularity(dict(zip(list(range(len(labels_pred7))), labels_pred7)), G, weight='weight')
result.append(['GEMSEC', len(set(labels_pred7)), modularity])
pd.DataFrame(result, columns=c).to_csv('./experiment_result/envo_baselines_real_full.csv')
print(7)

model8 = EdMot()
model8.fit(G)
cluster_membership8 = model8.get_memberships()
clusters8 = cluster_membership8
labels_pred8 = list(_flatten(list(clusters8.values())))
modularity = community.modularity(dict(zip(list(range(len(labels_pred8))), labels_pred8)), G, weight='weight')
result.append(['EdMot', len(set(labels_pred8)), modularity])
pd.DataFrame(result, columns=c).to_csv('./experiment_result/envo_baselines_real_full.csv')
print(8)

model9 = SCD()
model9.fit(G)
cluster_membership9 = model9.get_memberships()
clusters9 = cluster_membership9
labels_pred9 = list(_flatten(list(clusters9.values())))
modularity = community.modularity(dict(zip(list(range(len(labels_pred9))), labels_pred9)), G, weight='weight')
result.append(['SCD', len(set(labels_pred9)), modularity])
pd.DataFrame(result, columns=c).to_csv('./experiment_result/envo_baselines_real_full.csv')
print(9)

model10 = LabelPropagation()
model10.fit(G)
cluster_membership10 = model10.get_memberships()
clusters10 = cluster_membership10
labels_pred10 = list(_flatten(list(clusters10.values())))
modularity = community.modularity(dict(zip(list(range(len(labels_pred10))), labels_pred10)), G, weight='weight')
result.append(['LabelPropagation', len(set(labels_pred10)), modularity])
pd.DataFrame(result, columns=c).to_csv('./experiment_result/envo_baselines_real_full.csv')
print(10)

labels_pred11 = community.best_partition(G)
modularity = community.modularity(labels_pred11, G, weight='weight')
result.append(['Louvain', len(set(labels_pred11.values())), modularity])
pd.DataFrame(result, columns=c).to_csv('./experiment_result/envo_baselines_real_full.csv')
print(11)

print('Done!')