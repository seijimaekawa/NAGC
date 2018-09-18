#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NJNMF with PU-learning is described in this program.
We recommend using kmeans initialization (init=1).
In that case, you run kmeans_init.py in advance.
"""

import time
import sys
import scipy.stats
import numpy as np
import scipy as sp
import VU_init
# import networkx as nx
# from matplotlib import pyplot
# max_iter=100 #number of iterations
b=0.0   #bias for sigmoid function

class NAGC:
    def __init__(self,k1,k2,lam,S,W,X,data,init=1,rho=0.5,max_iter=100):
        node_size = S.shape[0]
        att_size = X.shape[1]
        #T is a matrix between topic1 and topic2
        H = np.random.random((k1,k2))
        #create initialized matrices
        U,V = VU_init.VU_init(X,k1,k2,init,data)
        # a = (1.0+rho)/2
        # W = np.abs(S_ori-1)
        W_ = 1-W
        # general params
        self.S = S
        self.W = W
        self.W_ = W_
        self.X = X
        self.max_iter = max_iter
        self.U = U
        self.H = H
        self.V = V
        self.lam = lam
        self.rho = rho
    def fit_predict(self):
        def update_U(S,W,W_,X,lam,U,H,V,rho):
            fUH = sigmoid(U.dot(H))
            fdUH = dif_sig(U.dot(H))
            UUT = U.dot(U.transpose())
            U = U*((rho*2.0)*S.dot(U)+(lam*X.dot(V) * fdUH).dot(H.transpose()))/((2.0*rho)*(UUT*W).dot(U)+(2.0*(1.0-rho))*(UUT*W_).dot(U)+(lam*fUH.dot(V.transpose().dot(V)) * fdUH).dot(H.transpose()))
            #V = V*((a*2.0)*S.dot(V)+(lam*A.dot(U) * fdVT).dot(T.transpose()))/(((2.0*a)*(VVT*S_ori)+(2.0*(1.0-a))*(VVT*S_)).dot(V)+(lam*fVT.dot(U.transpose().dot(U)) * fdVT).dot(T.transpose()))
            return U
        def update_U_woPU(S,X,lam,U,H,V):
            fUH = sigmoid(U.dot(H))
            fdUH = dif_sig(U.dot(H))
            U = U*(2.0*S.dot(U)+(lam*X.dot(V) * fdUH).dot(H.transpose()))/(2*U.dot(U.transpose().dot(U))+(lam*fUH.dot(V.transpose().dot(V)) * fdUH).dot(H.transpose()))
            return U
        def update_V(S,X,U,H,V):
            # fVT = sigmoid(V.dot(T))
            return V*(X.transpose().dot(sigmoid(U.dot(H))))/(V.dot(sigmoid(U.dot(H).transpose()).dot(sigmoid(U.dot(H)))))
        def update_H(S,X,U,H,V):
            fUH = sigmoid(U.dot(H))
            fdUH = dif_sig(U.dot(H))
            H = H*(U.transpose().dot(fdUH*(X.dot(V))))/(U.transpose().dot((fdUH*fUH).dot(V.transpose().dot(V))))
            return H
        def removing_nan(mat):
            nan_list = np.argwhere(np.isnan(mat))
            for i in nan_list:
                mat[i[0],i[1]]=sys.float_info.epsilon
            return mat
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-(x-b)))
        def dif_sig(x):
            return 1.0 * np.exp(-(x-b)) / pow(1.0+np.exp(-(x-b)),2)

        start = time.time() #memo start time
        #learning step
        count = 0
        while 1:
            count += 1
            # print loss_function(S,V,U,Z,A,T,lam)
            if self.rho == 0.5:
                self.U = removing_nan(update_U_woPU(self.S,self.X,self.lam,self.U,self.H,self.V))
            else:
                self.U = removing_nan(update_U(self.S,self.W,self.W_,self.X,self.lam,self.U,self.H,self.V,self.rho))
            self.V = removing_nan(update_V(self.S,self.X,self.U,self.H,self.V))
            self.H = removing_nan(update_H(self.S,self.X,self.U,self.H,self.V))
            if count>=self.max_iter:
                break
        elapsed_time = time.time() - start  #measure elapsed time
        print (("optimizing_time:{0}".format(elapsed_time)) + "[sec]")
        return self.U
