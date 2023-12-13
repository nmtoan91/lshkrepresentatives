import os
import os.path
import sys
from sys import platform

import numpy as np
import pandas as pd
#from kmodes_lib import KModes
from collections import defaultdict
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
import timeit
from kmodes.util import get_max_value_key, encode_features, get_unique_rows, \
    decode_centroids, pandas_to_numpy

from .ClusteringAlgorithm import ClusteringAlgorithm
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
import random
from .LSH import LSH 
import multiprocessing as mp
import math
from collections import defaultdict

class LSHkRepresentatives(ClusteringAlgorithm):

    def SetupLSH(self, hbits=-1,k=-1,measure='DILCA' ):
        #hbits = 2*math.ceil(math.log(self.k)/math.log(2))
        start = timeit.default_timer()
        self.lsh = LSH(self.X,self.y,measure=measure,hbits=hbits)
        self.lsh.DoHash()
        #self.lsh.CorrectSingletonBucket()
        
        self.time_lsh = timeit.default_timer() - start
        self.AddVariableToPrint("Time_lsh",self.time_lsh)
        return self.time_lsh
    def SetupMeasure(self, classname):
        self.measurename = classname
        module = __import__(classname, globals(), locals(), ['object'])
        class_ = getattr(module, classname)
        self.measure = class_()
        self.measure.setUp(self.X, self.y)
    def test(self):
        print("a234 " + str(self.k))
    def Distance(self,representative, point):
        sum=0
        for i in range (self.d):
            sum = sum + representative[i][point[i]]
        return self.d - sum
    
    def MovePoint(self, point_id, from_id, to_id ,representatives_count, representatives_sum,membship, curpoint,labels_matrix):
        labels_matrix[point_id] = to_id
        membship[to_id, point_id] = 1
        membship[from_id, point_id] = 0
        representatives_sum[to_id]+=1
        representatives_sum[from_id]-=1 
        for ii, val in enumerate(curpoint):
            representatives_count[to_id][ii][val]+=1
            representatives_count[from_id][ii][val]-=1
    def CheckEmptyClusters(self,representatives, X,representatives_sum, representatives_count,membship,labels_matrix):
        move =0
        big_cluster_id = -1
        for ki in range(self.k):
            if representatives_sum[ki] ==0 :
                #print("EMPTY: ", ki)
                big_cluster_id = np.argmax([sum(mem_) for mem_ in membship])
                choices = [i for i in range(self.n) if membship[big_cluster_id][i] == 1 ]
                #rindx = self.random_state.choice(choices)
                rindx = int(self.farest_point[big_cluster_id][0])
                self.MovePoint(rindx, big_cluster_id,ki, representatives_count, representatives_sum,membship,self.X[rindx],labels_matrix  )
                move +=1

        return move
    def InitClusters(self,representatives,representatives_sum,representatives_count):
        for ki in range(self.k):
            for i in range(self.d):
                sum_ = 0
                for j in range(self.D[i]): sum_ = sum_ + representatives[ki][i][j]
                for j in range(self.D[i]): representatives[ki][i][j] = representatives[ki][i][j]/sum_;

    def DistanceRepresentativestoAPoints(self,representatives, point):
        dist_matrix = [self.Distance(c, point) for c in representatives]
        representative_id = np.argmin(dist_matrix)
        return representative_id, dist_matrix[representative_id]

    def UpdateLabelsInit(self,representatives, X,representatives_sum, representatives_count,membship,labels_matrix):
        cost = 0
        move = 0
        self.preferList= defaultdict(set)
        for ipoint, curpoint in enumerate(X):
            representative_id,tmp = self.DistanceRepresentativestoAPoints(representatives, curpoint)
            cost += tmp
            labels_matrix[ipoint] = representative_id
            membship[representative_id, ipoint] = 1
            representatives_sum[representative_id]+=1
            for ii, val in enumerate(curpoint):
                representatives_count[representative_id][ii][val]+=1
            self.preferList[self.lsh.hash_values[ipoint]].add(labels_matrix[ipoint])
        self.CheckEmptyClusters(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
        self.dist_matrix_tmp = [1000000000 for i in range(self.k)]
        return cost ,move, 0
    def UpdateLabelsLast(self,representatives, X,representatives_sum, representatives_count,membship,labels_matrix):
        cost = 0
        self.preferList= defaultdict(set)


        for ipoint, curpoint in enumerate(X):
            representative_id,tmp = self.DistanceRepresentativestoAPoints(representatives, curpoint)
            cost += tmp
            membship[representative_id, ipoint] = 1
        return cost ,0, 0

    def DistanceRepresentativestoAPoints_LSH(self,item_id, point,labels_matrix,representatives):
        myset = self.preferList[self.lsh.hash_values[item_id]]
        dist_min = 1000000000
        dist_index =-1
        for i in myset:
            dist = self.Distance(representatives[i], point)
            if dist_min > dist:
                dist_min = dist 
                dist_index = i
        return dist_index, dist_min 
    def DistanceRepresentativestoAPoints_LSH2(self,item_id, point,labels_matrix,representatives):
        #myset = self.preferList[self.lsh.hash_values[item_id]]
        #myset = self.near_clusters[self.lsh_group[item_id]]
        myset = self.near_clusters[labels_matrix[item_id]]

        dist_min = 1000000000
        dist_index =-1
        for i in myset:
            dist = self.Distance(representatives[i], point)
            if dist_min > dist:
                dist_min = dist 
                dist_index = i
        return dist_index, dist_min 

    def UpdateLabels(self,representatives, X,representatives_sum, representatives_count,membship,labels_matrix):
        cost = 0
        move = 0
        for i in range(self.k):
            self.farest_point[i][1] = 0

        for ipoint, curpoint in enumerate(X):
            #representative_id,tmp = self.DistanceRepresentativestoAPoints_LSH(ipoint, curpoint,labels_matrix,representatives)
            representative_id,tmp = self.DistanceRepresentativestoAPoints(representatives, curpoint)
            #representative_id,tmp = self.DistanceRepresentativestoAPoints_LSH2(ipoint, curpoint,labels_matrix,representatives)
            if tmp > self.farest_point[representative_id][1]:
                self.farest_point[representative_id][1] = tmp
                self.farest_point[representative_id][0] = ipoint
            cost += tmp
            
            if membship[representative_id, ipoint]: continue
            old_clust = labels_matrix[ipoint]
            self.MovePoint(ipoint, old_clust,representative_id, representatives_count, representatives_sum,membship,curpoint,labels_matrix  )
            move +=1
        #Check empty clusters
        move  += self.CheckEmptyClusters(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
        return cost ,move, 0

    def UpdateRepresentatives(self,representatives,representatives_sum,representatives_count ) :  
        for ki in range(self.k):
            for di in range(self.d):
                    for vj in range(self.D[di]):
                        representatives[ki][di][vj] =  representatives_count[ki][di][vj]/representatives_sum[ki]
        return 0
    def GetLabels(self, membship):
        labels = np.empty(self.n, dtype=np.uint16)
        for ki in range(self.k):
            for i in range(self.n):
                if membship[ki][i]:
                    labels[i] = ki
        return labels
    def DoCluster(self,n_group=2):
        self.AddVariableToPrint("n_group",n_group)
        self.k = k = n_clusters = self.k
        self.farest_point = np.zeros((self.k,2))

        self.name = "LSHkRepresentatives"
        #print("LSHkRepresentatives start clustering")
        #Init varibles
        X = self.X
        
        self.n = n = self.X.shape[0];
        self.d = d = X.shape[1]
        self.D = D = [len(np.unique(X[:,i])) for i in range(d) ]

        all_labels = []
        all_costs = []
        start_time = timeit.default_timer()
        #self.n_init=1; # Force set self.n_init
        for init_no in range(self.n_init):
            self.random_state = check_random_state(None)
            membship = np.zeros((k, n), dtype=np.uint8)
            labels_matrix = np.empty(n, dtype=np.uint16)
            
            for i in range(n): labels_matrix[i] = 65535
            representatives_count = [[[0 for i in range(D[j])] for j in range(d)]for ki in range(k)]
            representatives_sum = [0 for ki in range(k)]
            last_cost = float('inf')

            representatives = [[[0 for i in range(D[j])] for j in range(d)] for ki in range(k)]
            buckets = [(k,len(self.lsh.hashTable[k])) for k in self.lsh.hashTable.keys()]
            buckets2 = sorted(buckets, key=lambda x: -x[-1])
            buckets_map_to_centroids = {}
            
            masterkeys=[]
            n_group = int(self.k/n_group)
            self.near_clusters = [[] for j in range(self.k)]
            self.lsh_group = [0 for i in range(self.n)]
            for i in range( self.k - len(buckets2)):
                buckets2.append((0,0))
            for i in range(self.k):
                masterkeys.append(buckets2[i][0])
                buckets_map_to_centroids[buckets2[i][0]] = i

            dist_from_master_to_other = [[ self.lsh.hammingDistance(keymaster,key) for key in self.lsh.hashTable.keys()] for keymaster in masterkeys  ]
            dist_from_master_to_master = [[ self.lsh.hammingDistance(keymaster,key) for key in masterkeys] for keymaster in masterkeys  ]

            count_remains = [n_group+1 for i in range(self.k) ]
            ddd= init_no%self.k; 
            #ddd= random.randint(0, self.k)
            ddd_end = ddd+self.k
            for ki_ in range(ddd,ddd_end):
                ki = ki_%self.k
                self.near_clusters[ki] = np.argsort(dist_from_master_to_master[ki])[0:n_group+1]
                if ki not in self.near_clusters[ki]:
                    self.near_clusters[ki][n_group]=ki
                for i in self.near_clusters[ki]:
                    count_remains[i] -=1
                    if count_remains[i] <=0:
                        for ki2 in range(ki,self.k):
                            dist_from_master_to_master[ki2][i] = float('inf')

            for key_id, key in enumerate(self.lsh.hashTable.keys()):
                nearest_key=-1
                nearest_dist = float('inf')
                for keymaster_id, value in enumerate(masterkeys):
                    d_temp = dist_from_master_to_other[keymaster_id][key_id]
                    if d_temp < nearest_dist: nearest_dist= d_temp; nearest_key = value;
                ki = buckets_map_to_centroids[nearest_key]
                #ki= random.randint(0, self.k-1 )
                for i in self.lsh.hashTable[key]:
                    labels_matrix[i] = ki
                    membship[ki][i]=1
                    representatives_sum[ki]+=1
                    for ii, val in enumerate(X[i]):
                        representatives_count[ki][ii][val]+=1
                    self.lsh_group[i] = ki

            self.CheckEmptyClusters(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
            self.UpdateRepresentatives(representatives,representatives_sum,representatives_count ) 
            for i in range(self.n_iter):
                self.iter = i
                cost , move, count_empty = self.UpdateLabels(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
                self.UpdateRepresentatives(representatives,representatives_sum,representatives_count ) 
                if last_cost == cost and move==0: 
                    last_cost = self.UpdateLabelsLast(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
                    #print("last_cost=", last_cost, "last_cost2=",last_cost2)
                    break 
                last_cost = cost
                #print ("Iter: ", i , " Cost:", cost, "Move:", move)
            labels = self.GetLabels(membship)
            all_costs.append(cost)
            all_labels.append(labels)
            
        best = np.argmin(all_costs)
        labels = all_labels[best]
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        self.labels = labels
        print("LSH time:", self.time_lsh ,"Score: ", all_costs[best] , " Time:", self.time_score)
        self.representatives= representatives
        return self.labels
        # Update representives
    def predict(self,x):
        dist_matrix = self.DistanceRepresentativestoAPoints(self.representatives, x)
        return dist_matrix[0]

     
def Test_Simple():
    DB = tulti.LoadSynthesisData(n=128,d=16,k=8,sigma_rate=0.1); 
    MeasureManager.CURRENT_DATASET = DB['name']
    MeasureManager.CURRENT_MEASURE = 'DILCA'

    print("\n\n############## LSHkRepresentatives ###################")
    lshkrepresentatives = LSHkRepresentatives(DB['DB'],DB['labels_'] )
    lshkrepresentatives.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    lshkrepresentatives.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
    lshkrepresentatives.DoCluster()
    lshkrepresentatives.CalcScore()

    print("\n\n############## KMODES ###################")
    kmodes = kModes(DB['DB'],DB['labels_'] )
    kmodes.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    kmodes.DoCluster()
    kmodes.CalcScore()

    print("\n\n############## kRepresentatives ###################")
    kmodes = kRepresentatives(DB['DB'],DB['labels_'] )
    kmodes.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    kmodes.DoCluster()
    kmodes.CalcScore()

def Test(): 
    MeasureManager.CURRENT_DATASET = 'balance-scale.csv'
    MeasureManager.CURRENT_MEASURE = 'Overlap'
    if TDef.data!='': MeasureManager.CURRENT_DATASET = TDef.data
    if TDef.measure!='': MeasureManager.CURRENT_MEASURE = TDef.measure
    if TDef.test_type == 'syn':
        DB = tulti.LoadSynthesisData(TDef.n,  TDef.d, TDef.k)
        MeasureManager.CURRENT_DATASET= DB['name']
    else:
        DB = tulti.LoadRealData(MeasureManager.CURRENT_DATASET)
    print("\n\n############## LSHkRepresentatives ###################")
    lshkrepresentatives = LSHkRepresentatives(DB['DB'],DB['labels_'] ,dbname=MeasureManager.CURRENT_DATASET ,k=TDef.k)
    lshkrepresentatives.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    lshkrepresentatives.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
    lshkrepresentatives.DoCluster()
    lshkrepresentatives.CalcScore()


def TestDatasets(): 
    for dbname in MeasureManager.DATASET_LIST:
        DB = tulti.LoadRealData(dbname)
        MeasureManager.CURRENT_DATASET = dbname
        MeasureManager.CURRENT_MEASURE = 'Overlap'
        print("\n\n############## LSHkRepresentatives ###################")
        alo = LSHkRepresentatives(DB['DB'],DB['labels_'],dbname=MeasureManager.CURRENT_DATASET )
        alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
        alo.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
        alo.DoCluster()
        alo.CalcScore()
def TestMeasures():
    table = MyTable()
    for measure in MeasureManager.MEASURE_LIST:
        MeasureManager.CURRENT_MEASURE = measure
        for dataset in MeasureManager.DATASET_LIST:
            MeasureManager.CURRENT_DATASET=dataset
            DB = tulti.LoadRealData(MeasureManager.CURRENT_DATASET)
            alo = LSHkRepresentatives(DB['DB'],DB['labels_'],dbname=MeasureManager.CURRENT_DATASET )
            alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
            alo.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
            alo.DoCluster()
            alo.CalcScore()
            alo.AddValuesToMyTable(table,measure == MeasureManager.MEASURE_LIST[0],dataset)
            table.SaveToExcelFolder("RESULT/r20200916",alo.name,[i+1 for i in range(0,1000)])

if __name__ == "__main__":
    X = np.array([[0,0,0],[0,1,1],[0,0,0],[1,0,1],[2,2,2],[2,3,2],[2,3,2]])
    kreps = LSHkRepresentatives(n_clusters=2,n_init=5) 
    kreps.fit(X)
    print()
    print(kreps.labels)
    print()

    print(kreps.predict(X[0]))
    print(kreps.predict(X[1]))
    print(kreps.predict(X[2]))
    print(kreps.predict(X[3]))
    print(kreps.predict(X[4]))
    print(kreps.predict(X[5]))
   
