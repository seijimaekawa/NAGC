#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import sys
import scipy.stats
import numpy as np
import scipy as sp
# import networkx as nx
# from matplotlib import pyplot
# max_iter=100 #number of iterations
b=0.0   #bias for sigmoid function

# def build_graph(path): # switch for two form of file
#     if "graphml" in path:
#         print "graphml"
#         S,A,clus = for_graphml(path)
#         return S,A,clus,0
#     else:
#         print "cite"
#         S,A,clus = for_cites_contents(path)
#         return S,A,clus,1

# def for_graphml(path):
#     parser = pg.GraphMLParser()
#     g = parser.parse(path)
#     node_size = len(g.nodes())
#     att_size = len(g.get_attributs())
#     att_list = [""]*att_size
#     count = 0
#     for i in g.get_attributs():
#         att_list[count] = str(i).split(" : ")[0]
#         count += 1
#     att_list = att_list[:-2]
#     #remove 'edgeid' and 'weight'
#     att_size = att_size-2
#     S = np.zeros((node_size,node_size))
#     A = np.zeros((node_size,att_size))
#     #fill the adjacency matrix and attribute matrix
#     for i in g.nodes():
#         for j in i.children():
#             S[i.id][j.id] = 1
#             S[j.id][i.id] = 1
#         for j in range(len(att_list)):
#             A[i.id][j] = i[att_list[j]]
#     S,A=preprocess(S,A)
#     return S,A,[] #scaling S and A
#
# def for_cites_contents(path):
#     node=[]
#     att_list=[]
#     clus=[]
# ############ download atttributes #################
#     infiles = glob.glob(path+'/*.content')
#     for infile in infiles:
#         with open(infile) as f:
#             while True:
#                 line = f.readline()
#                 if line == '':
#                     break
#                 tmp = line.split("\t")
#                 node.append(tmp[0])
#                 att_list.append((tmp[1:-1]))
#                 clus.append(tmp[-1].replace("\n",""))
#     print "number of nodes : ",
#     print len(node)
#     print "number of attributes : ",
#     print len(att_list[0])
#     # print len(clus)
# ############ download edges #################
#     edges=[]
#     infiles = glob.glob(path+'/*.cites')
#     split_flag=0
#     if "WebKB" in path:
#         split_flag=1
#     for infile in infiles:
#         with open(infile) as f:
#             while True:
#                 line = f.readline()
#                 if line == '':
#                     break
#                 line = line.replace("\n","")
#                 if split_flag==0:
#                     tmp = line.split("\t")
#                 if split_flag==1:
#                     tmp = line.split(" ")
#                 ind0 = node.index(tmp[0])
#                 ind1 = node.index(tmp[1])
#                 edges.append((ind0,ind1))
#     print "number of edges : ",
#     print len(edges)
#     node_size = len(node)
#     att_size = len(att_list[0])
#     S = np.zeros((node_size,node_size))
#     A = np.zeros((node_size,att_size))
#     for i in range(len(edges)):
#         S[edges[i][0]][edges[i][1]] = 1
#         S[edges[i][1]][edges[i][0]] = 1
#     for i in range(len(att_list)):
#         for j in range(len(att_list[0])):
#             A[i][j] = int(att_list[i][j])
#     S,A=preprocess(S,A)
#     return S,A,clus #scaling S and A

def graph_operation(k1,k2,lam,S,A,max_iter,heat_path):
    node_size = S.shape[0]
    att_size = A.shape[1]
    modularity_list = []
    #create initialized matrices
    V = np.random.random((node_size,k1))
    U = np.random.random((att_size,k2))
    #Z is a matrix diciding weight of attribute
    Z = np.zeros((att_size,att_size))
    #T is a matrix between topic1 and topic2
    T = np.random.random((k1,k2))
    #initialize by 1/m
    for i in range(att_size):
        Z[i][i] = 1.0/att_size
    #learning step
    count = 0
    while 1:
        # print count
        # print "loss = ",
        # print loss_function(S,V,U,Z,A,T,lam)
        V = removing_nan(update_V(S,V,U,Z,A,T,lam))
        U = removing_nan(update_U(S,V,U,Z,A,T,lam))
        T = removing_nan(update_T(S,V,U,Z,A,T,lam))
        Z = removing_nan(update_Z(S,V,U,Z,A,T,lam))
        #normalize Z
        sum_Z=0
        for i in range(len(Z)):
            sum_Z+=Z[i][i]
        # print sum_Z
        Z = Z/sum_Z
        if check_convergence(S,V,U,Z,A,T) == True:
            break
        elif count>=max_iter:
            break
        count += 1

    # np.savetxt(heat_path+'_'+str(k1)+'_'+str(k2)+'_V.csv',V,delimiter=',')
    # np.savetxt(heat_path+'_'+str(k1)+'_'+str(k2)+'_VT.csv',V.dot(T),delimiter=',')
    # np.savetxt(heat_path+'_'+str(k1)+'_'+str(k2)+'_f_VT.csv',sigmoid(V.dot(T)),delimiter=',')
    # np.savetxt(heat_path+'_'+str(k1)+'_'+str(k2)+'_T.csv',T,delimiter=',')
    # np.savetxt(heat_path+'_'+str(k1)+'_'+str(k2)+'_U.csv',U,delimiter=',')
    # np.savetxt(heat_path+'_'+str(k1)+'_'+str(k2)+'_Z.csv',Z,delimiter=',')
    return V,T
    # pyplot.plot(np.arange(0,max_iter), modularity_list)
    # pyplot.savefig("mod_nl{0}_{1}".format(k1,k2))
    # cal_entropy
# def preprocess(S,A):
#     S = S / sum(sum(S))
#     A = A / sum(sum(A))
#     return S,A

def removing_nan(mat):
    nan_list = np.argwhere(np.isnan(mat))
    for i in nan_list:
        mat[i[0],i[1]]=sys.float_info.epsilon
    return mat

def update_V(S,V,U,Z,A,T,lam):
    fVT = sigmoid(V.dot(T))
    fdVT = dif_sig(V.dot(T))
    V = V*(S.dot(V)+S.transpose().dot(V)+(lam*A.dot(Z.dot(U)) * fdVT).dot(T.transpose()))/(2*V.dot(V.transpose().dot(V))+(lam*fVT.dot(U.transpose().dot(U)) * fdVT).dot(T.transpose()))
    return V

def update_U(S,V,U,Z,A,T,lam):
    # fVT = sigmoid(V.dot(T))
    return U*(Z.dot(A.transpose()).dot(sigmoid(V.dot(T))))/(U.dot(sigmoid(V.dot(T).transpose()).dot(sigmoid(V.dot(T)))))

def update_T(S,V,U,Z,A,T,lam):
    fVT = sigmoid(V.dot(T))
    fdVT = dif_sig(V.dot(T))
    T = T*(V.transpose().dot(fdVT*(A.dot(Z.dot(U)))))/(V.transpose().dot((fdVT*fVT).dot(U.transpose().dot(U))))
    return T

def update_Z(S,V,U,Z,A,T,lam):
    upper_Z = A.transpose().dot(sigmoid(V.dot(T)).dot(U.transpose()))
    lower_Z = A.transpose().dot(A).dot(Z)
    for i in range(len(Z)):
        Z[i,i] = Z[i,i]*upper_Z[i,i]/lower_Z[i,i]
    return Z
def check_convergence(S,V,U,Z,A,T):
    # print "check_convergence"
    return False

def loss_function(S,V,U,Z,A,T,lam):
    loss = (np.linalg.norm(S-V.dot(V.transpose()))+lam*np.linalg.norm(A.dot(Z)-sigmoid(V.dot(T)).dot(U.transpose())))/2
    return loss

# def cal_modularity(clus,S,k):
#     a = [0]*k
#     e = np.zeros((k,k))
#     for i in range(k):
#         for j in range(k):
#             for l in clus[i]:
#                 for m in clus[j]:
#                     e[i][j]+=S[l][m]
#     e = e / sum(sum(S))
#     for i in range(k):
#         a[i] = sum(e[i][:])
#     Q=0
#     for i in range(k):
#         Q+=e[i][i]-a[i]*a[i]
#     # print Q
#     return Q
#
# def cal_entropy(clus,A,k):
#     E=0
#     for t in range(len(A[0])):
#         for j in range(k):
#             if len(clus[j])==0:
#                 continue
#             att=0
#             for n in clus[j]:
#                 if A[n][t]==0:
#                     att+=1
#             pk=[1.0*att/len(clus[j]),1.0-1.0*att/len(clus[j])]
#             # print pk
#             E += len(clus[j]) * scipy.stats.entropy(pk)
#     return E/len(A)/len(A[0])

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-(x-b)))
def dif_sig(x):
    return 1.0 * np.exp(-(x-b)) / pow(1.0+np.exp(-(x-b)),2)

# if __name__ == '__main__':
#     start = time.time() #memo start time
#     k_list = [7,10,15,20] #number of topic1
#     k2 = 7 #number of topic2
#     lam_list = [1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,0.5] #weight to balance structural/attribute fusion
#     max_iter=100
#     # path = "/Users/maekawaseiji/Google Drive/study/conference/PAKDD/2017/data-of-JWNMF/without ground truth/Disney/Disney.graphml"
#     path = "/Users/maekawaseiji/Google Drive/study/conference/PAKDD/2017/data-of-JWNMF/without ground truth/AmazonFail/AmazonFailNumeric.graphml"
#     # path = "/Users/maekawaseiji/Google ドライブ/study/conference/PAKDD/2017/data-of-JWNMF/without ground truth/Disney/Disney.graphml"
#     # path = "/Users/maekawaseiji/Google ドライブ/study/conference/PAKDD/2017/data-of-JWNMF/without ground truth/AmazonFail/AmazonFailNumeric.graphml"
#     # path = "/Users/maekawaseiji/Google Drive/study/conference/PAKDD/2017/data-of-JWNMF/with ground truth/WebKB"
#     S, A, true_clus, flag = build_graph(path)
#     for k1 in k_list:
#         print k1
#         for lam in lam_list:
#             mod_list=[]
#             for i in range(5):
#                 mod_list.append(graph_operation(k1,k2,lam,S,A,true_clus,flag,max_iter))
#             print lam
#             print np.average(mod_list)
#
#     elapsed_time = time.time() - start  #measure elapsed time
#     # print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
