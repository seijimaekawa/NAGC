#This program build adjacency matrix and attribute matrix from graph data.
#This program can deal with two format(.mat and .cite&.content)
#Outputs are blow
'''
S : adjacency matrix with prepocess
S_ori : original adjacency matrix
A : attribute matrix with preprocess
clus : true cluster of nodes
flag : with ground truth or without
A_ori : original attribute matrix
'''

import numpy as np
import glob
import scipy.io

def build_graph(path): # switch for two form of file
    if "mat" in path:
        print ("mat")
        S,S_ori,A,clus,A_ori = for_mat(path)
        flag=1
        if clus==[]:
            flag=0
        return S,S_ori,A,clus,flag,A_ori
    else:
        print ("cite")
        S,S_ori,A,clus,A_ori = for_cites_contents(path)
        return S,S_ori,A,clus,1,A_ori

# def for_mat(path):
#     matdata = scipy.io.loadmat(path)
#     a = matdata["A"]
#     f = matdata["F"]
#     node_size = a.shape[0]
#     att_size = f.shape[1]
#     # S = lil_matrix((node_size,node_size))
#     S = np.zeros((node_size,node_size))
#     A = np.zeros((node_size,att_size))
#     #fill the adjacency matrix and attribute matrix
#     nonzeros = a.nonzero()
#     print ("no.nodes: " + str(node_size))
#     print ("no.attributes: " + str(att_size))
#     edgecount=0
#     for i in range(len(nonzeros[0])):
#         S[nonzeros[0][i],nonzeros[1][i]] = 1
#         if nonzeros[0][i]<=nonzeros[1][i]:
#             edgecount+=1
#     print ("no.edges: " + str(edgecount))
#     # S=S.tocsr()
#     nonzeros = f.nonzero()
#     for i in range(len(nonzeros[0])):
#         A[nonzeros[0][i]][nonzeros[1][i]] = f[nonzeros[0][i],nonzeros[1][i]]
#     # print A
#     # reg=1
#     # A_copy = copy.deepcopy(A)
#     # if reg == 1:
#     #     for i in range(A_copy.shape[1]):
#     #         max_att = max(A_copy[:,i])
#     #         if max_att != 0:
#     #             A_copy[:,i] = A_copy[:,i]/max_att*100
#     S_pre,A_pre=preprocess(S,A)
#     return S_pre,S,A_pre,[],A #scaling S and A

def for_mat(path):
    mat_contents = scipy.io.loadmat(path)
    G = mat_contents["Network"]
    X = mat_contents["Attributes"]
    Label = list(map(int,mat_contents["Label"]))
    node_size = G.shape[0]
    att_size = X.shape[1]
    # S = lil_matrix((node_size,node_size))
    # S = np.zeros((node_size,node_size))
    # A = np.zeros((node_size,att_size))
    S = np.zeros((node_size,node_size))
    A = X.toarray()
    #fill the adjacency matrix and attribute matrix
    nonzeros = G.nonzero()
    print ("no.nodes: " + str(node_size))
    print ("no.attributes: " + str(att_size))
    edgecount=0
    for i in range(len(nonzeros[0])):
        S[nonzeros[0][i],nonzeros[1][i]] = 1
        S[nonzeros[1][i],nonzeros[0][i]] = 1
    # erase diagonal element
    diag = 0
    for i in range(node_size):
        diag += S[i,i]
    nonzeros = S.nonzero()
    edge_count = int((len(nonzeros[0])+diag)/2)
    print ("number of edges : " + str(edge_count))

    S_pre,A_pre=preprocess(S,A)
    return S_pre,S,A_pre,Label,A #scaling S and A


def for_cites_contents(path):
    node={}
    counter=0
    att_list=[]
    clus=[]
############ download atttributes #################
    infiles = glob.glob(path+'/*.content')
    for infile in infiles:
        with open(infile) as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                tmp = line.split("\t")
                node[tmp[0]]=counter
                counter+=1
                att_list.append((tmp[1:-1]))
                clus.append(tmp[-1].replace("\n",""))
    print ("number of nodes : " + str(len(node)))
    print ("number of attributes : " + str(len(att_list[0])))
    # print (clus)
############ download edges #################
    edges=[]
    infiles = glob.glob(path+'/*.cites')
    for infile in infiles:
        with open(infile) as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                line = line.replace("\n","")
                tmp = line.split()
                if tmp[0] in node and tmp[1] in node:
                    ind0 = node[tmp[0]]
                    ind1 = node[tmp[1]]
                    edges.append((ind0,ind1))
    node_size = len(node)
    att_size = len(att_list[0])
    # S = lil_matrix((node_size,node_size))
    S = np.zeros((node_size,node_size))
    # A = lil_matrix((node_size,att_size))
    A = np.zeros((node_size,att_size))
    for i in range(len(edges)):
        S[edges[i][0],edges[i][1]] = 1
        S[edges[i][1],edges[i][0]] = 1
    # erase diagonal element
    diag = 0
    for i in range(node_size):
        diag += S[i,i]
    nonzeros = S.nonzero()
    edge_count = int((len(nonzeros[0])+diag)/2)
    print ("number of edges : " + str(edge_count))
    # S=S.tocsr()
    for i in range(len(att_list)):
        for j in range(len(att_list[0])):
            A[i,j] = float(att_list[i][j])
    # A=A.tocsr()

    S_pre,A_pre=preprocess(S,A)
    return S_pre,S,A_pre,clus,A #scaling S and A

def preprocess(S,A):
    # initialization in the paper(JWNMF)
    # S = S / S.sum()
    # A = A / A.sum()

    # initialization based on size of S
    # A = A * S.sum() / A.sum()
    S = S * A.sum() / S.sum()
    return S,A
