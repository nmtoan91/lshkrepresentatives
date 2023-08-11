import numpy as np

import math
import os
import os.path
import sys
from sys import platform
sys.path.append(os.path.join(os.getcwd(), "Measures"))
import networkx as nx
from collections import defaultdict
import math

class SimpleHashing:
    ALGORITHM_LIST = ['LSH']
    def __init__(self, X, y,n_init=10,hbits=-1,k=-1,measure='DILCA'):
        self.X = X
        self.y = y
        self.n = len(self.X)
        self.d = len(self.X[0])
        self.D = D = [len(np.unique(self.X[:,i])) for i in range(self.d) ]
        self.k = k if k > 0 else len(np.unique(y))
        self.hbits = hbits if hbits > 0 else int( math.log(self.n*2/ len(np.unique(y)))/ math.log(2))
        
        while 2**(self.hbits-1) < self.k:
            self.hbits +=1
        if(self.hbits>=self.d) : self.hbits = self.d-1

        if 2**self.hbits <=self.k: print("WARNING: BAD HBITS: hbits=",self.hbits, " d=", self.d,  " nbucket=", 2**self.hbits," k=", self.k, ' n=',self.n)
        if n_init > 0: self.n_init = n_init
        self.SetupMeasure(measure)
    def SetupMeasure(self, classname):
        module = __import__(classname, globals(), locals(), ['object'])
        class_ = getattr(module, classname)
        self.measure = class_()
        self.measure.setUp(self.X, self.y)
    def SetupMeasure(self, classname):
        #module = __import__(classname, globals(), locals(), ['object'])
        #class_ = getattr(module, classname)
        if classname=='DILCA':
            from .DILCA import DILCA
            class_ = DILCA
        self.measure = class_()
        self.measure.setUp(self.X, self.y)
    def Overlap(self,x,y):
        n = len(x)
        sum =0
        for i in range(n):
            if x[i] != y[i]: sum +=1
        return sum
    def test(self):
        print("n=",self.n, " d=",self.d, ' hbit=', self.hbits, " k=",self.k , " measure=", self.measure)
    #def DoHash
    def GenerateSimilarityMatrix(self, simMatrix):
        self.partitions = [];
        self.cut_values =[]
        self.cut_values_normal =[]
        for di in range(self.d):
            G = nx.Graph()
            for i in range(self.D[di]):
                matrix1D = [] # 1D array for 1 dimension
                for j in range(i+1,self.D[di]): 
                    #simMatrix[di][i][j]
                    G.add_edge(i,j,weight=simMatrix[di][i][j])
            if len(G.nodes)>1:
                ut_value, partition = nx.stoer_wagner(G)
                self.partitions.append(partition)
                self.cut_values.append(ut_value)
                self.cut_values_normal.append(ut_value/ self.D[di])
            else :
                self.partitions.append([[0],[0]])
                self.cut_values.append(10000000)
                self.cut_values_normal.append(10000000)