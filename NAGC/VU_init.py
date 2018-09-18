#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random

def VU_init(A,k1,k2,flag,data):
    n = A.shape[0]
    m = A.shape[1]
    if flag == 0:
        U = np.random.random((n,k1))
        V = np.random.random((m,k2))
    else:
        infile = 'initialize/'+data+'_U_'+str(k1)+'.csv'
        U = []
        with open(infile) as f:
            while True:
                row = []
                line = f.readline()
                if line == '':
                    break
                line = line.rstrip('\n')
                tmp = line.split(',')
                for i in tmp:
                    row.append(float(i))
                U.append(row)
        infile = 'initialize/'+data+'_V_'+str(k2)+'.csv'
        V = []
        with open(infile) as f:
            while True:
                row = []
                line = f.readline()
                if line == '':
                    break
                line = line.rstrip('\n')
                tmp = line.split(',')
                for i in tmp:
                    row.append(float(i))
                V.append(row)
        U = np.array(U)
        V = np.array(V).transpose()
    return U,V
