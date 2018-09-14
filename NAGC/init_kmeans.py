#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
import argparse
import build_graph
import numpy as np
import csv
import evaluate

def clustering(A,k):
    kmeans_clus=[]
    for i in range(k):
        kmeans_clus.append([])
############# regularization for input attributes
    for i in range(A.shape[1]):
        max_att = max(A[:,i])
        if max_att != 0:
            A[:,i] = A[:,i]/max_att
############ Learing step of kmeans
    kmeans = KMeans(n_clusters=k).fit(A)
    pred = kmeans.labels_
    centroids = kmeans.cluster_centers_
    for i in range(len(pred)):
        kmeans_clus[pred[i]].append(i)
    return pred, centroids, kmeans_clus
def initialize_V(A, centroids):
    V = np.zeros((len(A),len(centroids)))
    for i in range(V.shape[0]):
        dis_list = []
        for j in range(V.shape[1]):
            dis_list.append(np.linalg.norm(A[i]-centroids[j]))
        for j in range(V.shape[1]):
            V[i,j]= (sum(dis_list)-dis_list[j]) / sum(dis_list)
        V[i,:] = V[i,:] / sum(V[i,:])
    return V

#if __name__ == '__main__':
#    parser = argparse.ArgumentParser(
#                prog='argparseTest', # プログラム名
#                usage='Demonstration of argparser', # プログラムの利用方法
#                description='need filename and the number of clusters', # 引数のヘルプの前に表示
#                epilog='end', # 引数のヘルプの後で表示
#                add_help=True, # -h/–help オプションの追加
#                )
#    parser.add_argument('-k', help='integer', type=int, default=1)
#    parser.add_argument('-name', type=str, default="")
#    parser.add_argument('-ent', type=int, default=0) # 属性に連続値を扱う場合は1を指定
#    args = parser.parse_args()
    
def init_kmeans(k,data):
    print(data)

    path = "data/"+data
    S, S_ori, A, true_clus, flag, A_ori = build_graph.build_graph(path)
    clus_list = list(set(true_clus))
    print(clus_list)
    clus_dic = {}
    for i in range(len(clus_list)):
        clus_dic[clus_list[i]]=i
    for i in range(len(true_clus)):
        true_clus[i] = clus_dic[true_clus[i]]

    pred_l=[];cent_l=[];km_l=[]
    mod=[];ent=[];nmi=[];ari=[]
    for j in range(5):
        pred, centroids, kmeans_clus = clustering(A_ori,k)
        pred_l.append(pred)
        cent_l.append(centroids)
        km_l.append(kmeans_clus)
        ari.append(evaluate.ARI(true_clus,pred))
    ind=ari.index(sorted(ari)[2])
    pred=pred_l[ind]
    centroids=cent_l[ind]
    kmeans_clus=km_l[ind]
    V = initialize_V(A_ori, centroids)
    f = open('initialize/'+data+'_V_'+str(k)+'.csv','w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(V)
    f.close()
    f = open('initialize/'+data+'_U_'+str(k)+'.csv','w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(centroids)
    f.close()