import timeit
import os
import os.path
import  csv
import json
from .MeasureManager import MeasureManager
import numpy as np
from termcolor import colored
class Measure(object):
    
    def calculate(self):
        return 'Finish caculating'
    def LoaddistMatrixAuto(self):
        if MeasureManager.IS_LOAD_AUTO == False or MeasureManager.CURRENT_DATASET=="None":
            print( 'SKIP LOADING distMatrix because: ' + str(MeasureManager.IS_LOAD_AUTO) +" bd=" +MeasureManager.CURRENT_DATASET )
            return False
        path = 'saved_dist_matrices/json/' + self.name+"_" + MeasureManager.CURRENT_DATASET+ ".json"
        if os.path.isfile(path):
            with open(path, "r") as fp:
                self.distMatrix = json.load(fp)
        else: 
            print(colored('CANNOT OPEN FILE: ' +path),'yellow' )
            return False

        print("Loaded dist matrix: " , path)
        return True
    def SavedistMatrix(self):
        if not os.path.exists('saved_dist_matrices'):
            os.makedirs('saved_dist_matrices')
        if not os.path.exists('saved_dist_matrices/json'):
            os.makedirs('saved_dist_matrices/json')
        path = 'saved_dist_matrices/json/' + self.name+"_" + MeasureManager.CURRENT_DATASET+ ".json"
        with open('saved_dist_matrices/' + self.name+"_" + MeasureManager.CURRENT_DATASET, 'w') as f:
            wr = csv.writer(f)
            wr.writerows(self.distMatrix)
        with open(path, "w") as fp:
            json.dump(self.distMatrix, fp)
        print("Saving",self.name,"to:" ,path)
    def setUp(self, X, y):
        self.X_ = X
        self.y_ = y
        return 0
    def test(self):
        print('Test OK')
    def GeneratedistMatrix(self):
        D = len(self.X_[0])
        self.max = []
        for i in range(len(self.X_[0])):
            self.max.append(max(self.X_[:,i]))
        self.distMatrix = [];
        for d in range(D):
            matrix2D = [] # 2D array for 1 dimension
            for i in range(self.max[d]+1):
                matrix1D = [] # 1D array for 1 dimension
                for j in range(self.max[d]+1): 
                    matrix_tmp = self.CalcdistanceArrayForDimension(d,i,j)
                    matrix1D.append(matrix_tmp)
                matrix2D.append(matrix1D)
            self.distMatrix.append(matrix2D)
    def GeneratesimMatrix(self):
        self.d = d = len(self.X_[0])
        self.simMatrix = [];
        self.D = D = [len(np.unique(self.X_[:,i])) for i in range(d) ]
        for di in range(d):
            matrix2D = [] # 2D array for 1 dimension
            for i in range(D[di]):
                matrix1D = [] # 1D array for 1 dimension
                for j in range(D[di]):
                    #matrix_tmp = 1-self.distMatrix[di][i][j]
                    if self.distMatrix[di][i][j] ==0: matrix_tmp = 10000;
                    else : matrix_tmp= 1/self.distMatrix[di][i][j]

                    matrix1D.append(matrix_tmp)
                matrix2D.append(matrix1D)
            self.simMatrix.append(matrix2D)
