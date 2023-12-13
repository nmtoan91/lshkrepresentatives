from .Measure import Measure
import math
from .MathUtils import *
from scipy.stats import entropy
import numpy as np
import timeit

class DILCA(Measure):
    def __init__(self):
        self.name = "DILCA"
    def ProconditionMatrixYX(self, Y, X,Y_unique, X_unique):
        lenData = len(Y)
        lenX = len(X_unique)
        lenY = len(Y_unique)
        HY = 0
        count_x = 0;
        count_y=0;
        MATRIX = [[0 for i in range(lenX)] for j in  range(lenY)]
        for x in range(lenX):
            for y in range(lenY):
                count_x = count_y =0
                for i in range(lenData):
                    if X[i] == x:
                        count_x = count_x+1
                        if(Y[i] == y):
                            count_y = count_y+1;
                MATRIX[y][x] = count_y/count_x if count_x > 0 else 0
                asd = 12332
        return MATRIX

    def ComputeEtropy(self,X):
        N = len(X);
        D = len(X[0]);
        context_array = [[] for _ in range(D)]
        self.contextMatrix = []
        self.conditionProMatrix= {}
        self.probabilityMatrix = [];
        #Compute cross contextMatrix 
        for i in range(D):
            Y_ = X[:,i]
            Y_unique, Y_freq = np.unique(Y_, return_counts=True)
            X_probability  = [ii/len(Y_) for ii in Y_freq]
            self.probabilityMatrix.append(X_probability)
            HY = entropy(X_probability,base=2)
            SUY_dict = {}
            for j in range(D):
                if i !=j:
                    X_ = X[:,j]
                    X_unique, X_freq = np.unique(X_, return_counts=True)
                    X_property = [ii/len(X_) for ii in X_freq]
                    HX = entropy(X_property,base=2)
                    conditionMatrix = self.ProconditionMatrixYX(Y_,X_,Y_unique, X_unique)
                    self.conditionProMatrix[(i,j)] = conditionMatrix
                    HYX = 0
                    for k in range(len(X_unique)):
                        sum_tmp =0;
                        for k2 in range(len(Y_unique)):
                            if conditionMatrix[k2][k] !=0:
                                sum_tmp = sum_tmp +  conditionMatrix[k2][k] * math.log2(conditionMatrix[k2][k])
                        HYX = HYX + sum_tmp*X_property[k]
                    HYX= -HYX 
                    IGYX = HY - HYX
                    if(HX + HY ==0):
                        SUYX=0
                    else:
                        SUYX = 2*IGYX/(HY + HX)
                    SUY_dict[j] = SUYX
            values = list(SUY_dict.values())
            o = 1
            mean = np.mean(values)
            context_Y = [ key for (key,value) in  SUY_dict.items() if value >= o*mean]
            self.contextMatrix.append(context_Y)
        #Compute distMatrix
        self.distMatrix = [];
        for d in range(D):
            matrix = [] # 2D array for 1 dimension
            for i in range(self.max[d]+1):
                matrix_tmp = []; #1D array for 1 values on the attribute d
                for j in range(self.max[d]+1):
                    dist_sum_all =0;
                    dist_sum = 0;
                    dist_sum2 = 0;
                    for d2 in self.contextMatrix[d]:
                        dist_sum_tmp =0;
                        conditionMatrix = self.conditionProMatrix[(d,d2)]
                        for i_k in range(self.max[d2]):
                            dist_sum_tmp2 =  math.pow(conditionMatrix[i][i_k] -  conditionMatrix[j][i_k],2)
                            dist_sum_tmp = dist_sum_tmp +  dist_sum_tmp2*self.probabilityMatrix[d2][i_k];
                        dist_sum = dist_sum+ dist_sum_tmp
                        dist_sum2 = dist_sum2 + self.max[d2]+1
                    if dist_sum2==0: # toanstt 
                        dist_sum2=1;
                    dist_sum_all = math.sqrt(dist_sum/dist_sum2)
                    matrix_tmp.append(dist_sum_all)
                matrix.append(matrix_tmp)
            self.distMatrix.append(matrix)
    def calculate(self, instance1, instance2):
        distance = 0
        length = len(instance1)
        for x in range(length):
            distance =distance+ math.pow(self.distMatrix[x][instance1[x]][instance2[x]],2)
        return math.sqrt(distance)

    def setUp(self, X, y):
        start = timeit.default_timer()
        self.max = []
        self.X_ = X
        if self.LoaddistMatrixAuto()==False:
            print("Generating disMatrix for DILCA")
            for i in range(len(X[0])):
                self.max.append(max(X[:,i]))
            self.ComputeEtropy(X)
            self.SavedistMatrix()
        return timeit.default_timer() - start

