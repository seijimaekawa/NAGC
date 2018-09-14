#!/usr/bin/env python
# -*- coding: utf-8 -*-
#this program include calculater of modularity, entropy, NMI and ARI
import sys
import scipy.stats
import numpy as np
import scipy as sp
import scipy.io
from scipy.sparse import lil_matrix, csr_matrix
import math
from sklearn.metrics.cluster import adjusted_rand_score

def evaluate(mod,ent,spl,nmi,ari,true_clus,clus,pred,S,A,ent_flag,with_gt):
    k = len(clus)
    mod.append(cal_modularity(clus,S,k))
    if ent_flag == 0:
        ent.append(cal_entropy(clus,A,k))
    elif ent_flag == 1:
        ent.append(cal_dif_entropy(clus,A_copy,k))
        spl.append(split_entropy(clus,A_ori,k))
    if with_gt >= 3:
        nmi.append(cal_nmi(true_clus,clus))
        ari.append(ARI(true_clus,pred))
    return mod,ent,spl,nmi,ari
#this function calculates modularity from clustering result
def cal_modularity(clus,S,k):
    a = [0]*k
    e = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            for l in clus[i]:
                for m in clus[j]:
                    e[i][j]+=float(S[l,m])
    # regularize e on the sum of S
    e = e / S.sum()
    for i in range(k):
        a[i] = sum(e[i][:])
    Q=0
    for i in range(k):
        Q+=e[i][i]-a[i]*a[i]
    return Q

def cal_entropy(clus,A,k):
    E=0
    for t in range(A.shape[1]):
        for j in range(k):
            if len(clus[j])==0:
                continue
            att=0
            for n in clus[j]:
                if A[n,t]==0:
                    att+=1
            pk=[1.0*att/len(clus[j]),1.0-1.0*att/len(clus[j])]
            # print pk
            E += len(clus[j]) * scipy.stats.entropy(pk)
    return E/A.shape[0]/A.shape[1]

def cal_nmi(true_clus, clus):
    t_label_list = list(set(true_clus))
    n_h=[]
    for i in range(len(t_label_list)):
        n_h.append(true_clus.count(i))
    t_clus_ind = [[] for i in range(len(n_h))]
    for i in range(len(true_clus)):
        t_clus_ind[true_clus[i]].append(i)
    n_l = []
    for i in range(len(clus)):
        n_l.append(len(clus[i]))

########### print No. labels ###############
    # print("true_No.label : [ "),
    # for i in range(len(n_h)):
    #     print(str(n_h[i])+" "),
    # print(']')
    print("estimate_No.label : [ ",end="")
    for i in range(len(n_l)):
        print(str(n_l[i])+" ",end="")
    print(']')
#############################################
    n=sum(n_l)
    nmi = 0
    #n_h : 正解集合hの文書数
    #n_l : クラスタlの文書数
    #n_hl : クラスタl中で正解集合hに属する文書数
    for h in range(len(n_h)):
        for l in range(len(n_l)):
            n_hl = len(list(set(t_clus_ind[h]) & set(clus[l])))
            if n_hl != 0 and n_h[h] != 0 and n_l[l] != 0:
                nmi += n_hl * math.log(1.0*n*n_hl/n_h[h]/n_l[l],2)
    deno_h = 0
    for h in range(len(n_h)):
        if n_h[h] != 0:
            deno_h += n_h[h] * math.log(1.0*n_h[h]/n,2)
    deno_l = 0
    for l in range(len(n_l)):
        if n_l[l] != 0:
            deno_l += n_l[l] * math.log(1.0*n_l[l]/n,2)
    if deno_h*deno_l != 0.0:
        nmi = nmi/math.sqrt(deno_h*deno_l)
    else:
        nmi = 0.0
    return nmi

def ARI(true_clus,clus):
    return adjusted_rand_score(true_clus,clus)
