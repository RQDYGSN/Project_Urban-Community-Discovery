# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:07 2020

@author: Liu Qinghe
"""
import geopandas as gpd
import pandas as pd 
from sklearn import preprocessing 
import networkx as nx
import numpy as np
import math
import random
import community
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modularity_maximization import partition
from modularity_maximization.utils import get_modularity
from modutils import mod_calc
assert torch.cuda.is_available()
device = torch.device('cuda')
gpus = [0]
from typing import Dict
from scipy.sparse import coo_matrix

class CDRAE(nn.Module):
    def __init__(self, A_hat, num_feat, num_hidden, W_prior, weight_list):
        super(CDRAE, self).__init__()
        f = nn.Softmax(dim=2)
        self.graph_num = A_hat.shape[0]
        self.num_feat = num_feat # 特征数 f
        self.num_hidden = num_hidden # 隐含数 h
        self.weight_list = weight_list
        self.A_hat = A_hat
        self.d = self.A_hat.sum(axis=1).float().contiguous().view(self.graph_num, -1, 1)
        self.dT = self.A_hat.sum(axis=2).float().contiguous().view(self.graph_num, -1, 1)
        self.ddT = torch.mul(self.d.expand(self.graph_num, self.num_feat, self.num_feat), self.dT.expand(self.graph_num, self.num_feat, self.num_feat))
        self.B = (self.A_hat - torch.div(self.ddT, torch.sum(torch.sum(self.A_hat, axis=-1), axis=-1).view(-1, 1, 1))).float()
        self.B_norm = torch.div(self.B, torch.sum(torch.sum(self.A_hat, axis=-1), axis=-1).view(-1, 1, 1))
        self.W_0 = nn.Parameter(f(torch.ones(self.graph_num, self.num_feat, self.num_hidden).to(device)+W_prior)) # [1][f*h]

    def forward(self, A_hat,temp, epoch): # X貌似暂时没有用到
        global featureSelector # 
        global weight_feature # 
        s = nn.Softmax(dim=2)
        featureSelector = self.W_0.clone() # [1][f*h]
        weight_feature_multi = s(featureSelector) # [-][f*h]
        weight_feature = torch.mean(weight_feature_multi, axis=0)

        H = torch.mm(torch.mm(weight_feature.T, torch.mean(self.B_norm*self.weight_list.view(-1,1,1), axis=0)), weight_feature) # weight_feature 就是我们要求的U
        H = torch.div(H, H.sum(axis=0)) # 按照每一列进行归一化，列为axis=0的方向。
        m = nn.Softmax(dim=0)
        return m(H)
        return H
    
def lossFn(output): 
    return torch.trace(-torch.log(output))

def th_kl_div(p, q):
    return torch.mean(torch.sum((0.00001+p)*torch.log(torch.div(0.00001+p, 0.00001+q)), axis=1))*p.shape[1]

def main(clusters_number, feature, multi_graph, alpha_1, alpha_2, alpha_3, weight_list, alpha, epoch_num):
    # clusters_number = 17
    # feature = feature
    # multi_graph = [G1]
    # alpha_1 = 1
    # alpha_2 = 0.1
    # alpha_3 = 0.01
    # weight_list = [1, 1]
    # alpha = 2
    
    W_prior_list = []
    for G in multi_graph:
        partition = community.best_partition(G)
        partition_count = {}
        for key, value in partition.items():
            if value not in partition_count:
                partition_count[value] = 1
            else:
                partition_count[value] = partition_count[value] + 1

        # print(community.modularity(partition, G, weight='weight'))
        remapping = {}
        idx = 0
        for key, value in partition_count.items():
            if value>1 and key not in remapping:
                remapping[key] = idx
                idx = idx + 1

        tmp = np.zeros((len(G), clusters_number))
        for key, value in partition.items():
            try:
                tmp[key, remapping[value]] = alpha
            except:
                pass

        W_prior_list.append(tmp)

    for i in range(len(W_prior_list)):
        W_prior_list[i] = torch.tensor(W_prior_list[i]).float().unsqueeze(0)
    W_prior = torch.cat(tuple(W_prior_list), 0).to(device)
    
    weight_list = torch.tensor(weight_list).float()
    weight_list = torch.div(weight_list, torch.sum(weight_list))
    weight_list = weight_list.to(device)

    num_feat = len(multi_graph[0].nodes()) # f = 节点数
    num_hidden = clusters_number # h = 聚类数

    A_list = []
    for i in multi_graph:
        A_list.append(torch.tensor(nx.adjacency_matrix(i).todense(), dtype=np.float).unsqueeze(0))
    A_hat = A_hat = torch.cat(tuple(A_list), 0) # a matrix: A
    A_hat_tensor = torch.Tensor(A_hat.float()).to(device)

    vector_beta = torch.ones((feature.shape[1], 1)).float().to(device)
    vector_omega = 1.0/clusters_number*torch.ones((clusters_number, 1)).float().to(device)
    feature = torch.tensor(feature).float().to(device)

    model = CDRAE(A_hat_tensor, num_feat, num_hidden, W_prior, weight_list).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    temp = 3
    for epoch in range(epoch_num):
        model.train()
        model.zero_grad()
        if(epoch == 75):
            temp = 2.75
        elif(epoch == 100):
            temp = 2.5
        elif(epoch == 125):
            temp = 2
        elif(epoch == 150):
            temp = 1.8
        elif(epoch == 175):
            temp = 1.25
        elif(epoch == 250):
            temp = 1.00
        elif(epoch == 300):
            temp = 0.75
        elif(epoch == 320):
            temp = 0.50
        elif(epoch == 400):
            temp = 0.20
        output = model(A_hat_tensor, temp, epoch)
        equality_o = torch.mm(torch.mm(weight_feature.T, feature),vector_beta)
        equality = equality_o/equality_o.sum()
        if epoch%10 == 0:
            P = torch.div(torch.square(weight_feature), torch.square(weight_feature).sum(axis=1).view(-1, 1))
        loss_clus = th_kl_div(weight_feature, P)
        # loss_pena = torch.sum(torch.mean(torch.square(featureSelector - torch.mean(featureSelector, axis=0).unsqueeze(0)), axis=0))
        loss_cons = torch.sqrt(torch.mean(torch.abs(equality - vector_omega))) # th_kl_div(equality, vector_omega)
        loss = alpha_1*lossFn(output) + alpha_2*loss_cons + alpha_3*loss_clus

        gumbel_matrix = weight_feature.clone().detach().max(dim=1)[1] # 每个节点的标签组成的tensor
        labels_pred = gumbel_matrix.cpu().data.numpy()
        ddd = dict(zip(list(range(len(labels_pred))), labels_pred))

#         if epoch%2==0: 
#             mod = 0
#             for aa in range(len(multi_graph)):
#                 mod_one = community.modularity(ddd, multi_graph[aa], weight='weight')
#                 mod = mod + weight_list[aa].item()*mod_one
#             print('Epoch: ', epoch, '; soft_mod: ', round(mod, 6), '; loss_cons: ', round(loss_cons.item(), 6), '; loss_clus: ', round(loss_clus.item(), 6))

        loss.backward(retain_graph=True)
        optimizer.step()
    
    ######################################################################################################
    gumbel_matrix = weight_feature.clone().detach().max(dim=1)[1] # 每个节点的标签组成的tensor
    labels_pred = gumbel_matrix.cpu().data.numpy()
    return dict(zip(list(range(len(labels_pred))), list(labels_pred)))