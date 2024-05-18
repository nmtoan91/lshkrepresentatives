import os
import os.path
import sys
from sys import platform

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
import timeit
# from kmodes.util import get_max_value_key, encode_features, get_unique_rows, \
#     decode_centroids, pandas_to_numpy

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
from .LSHkRepresentatives import CategoricalDatasetNormalization

        
class LSHkPrototypes(ClusteringAlgorithm):

    def SetupLSH(self, hbits=-1,k=-1,measure='DILCA' ):
        self.CheckNormalizedData()
        start = timeit.default_timer()
        self.lsh = LSH(self.X_CATE,self.y,measure=measure,hbits=hbits)
        self.lsh.DoHash()
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

    def Distance_CATE(self,representative, point):
        sum=0
        for i in range (self.d_CATE):
            sum = sum + representative[i][point[i]]
        return self.d_CATE - sum
    def Distance_NUM(self,center, point):
        return np.linalg.norm(center-point)

    def Distance_2(self,representative,mean, pointCATE, pointNUM):
        sum=0
        for i in range (self.d_CATE):
            sum = sum + representative[i][pointCATE[i]]
        sum =  self.d_CATE - sum

        return np.linalg.norm(pointCATE-pointNUM)*self.weight_NUM + sum*self.weight_CATE

    
    def MovePoint(self, point_id, from_id, to_id ,representatives_count, representatives_sum,membship, curpoint,labels_matrix,means_count,means_sum):
        labels_matrix[point_id] = to_id
        membship[to_id, point_id] = 1
        membship[from_id, point_id] = 0

        representatives_sum[to_id]+=1
        representatives_sum[from_id]-=1 

        for ii, val in enumerate(curpoint):
            representatives_count[to_id][ii][val]+=1
            representatives_count[from_id][ii][val]-=1


        means_count[to_id] += 1
        means_count[from_id] -= 1

        if  len(means_sum.shape)>1:
            for di in range(self.d_NUM):
                means_sum[to_id][di] += self.X_NUM[point_id][di]
                means_sum[from_id][di] -= self.X_NUM[point_id][di]

    def CheckEmptyClusters(self,representatives, X,representatives_sum, representatives_count,membship,labels_matrix,means_count,means_sum,means):
        move =0
        big_cluster_id = -1
        for ki in range(self.k):
            if representatives_sum[ki] ==0 :
                big_cluster_id = np.argmax([sum(mem_) for mem_ in membship])
                choices = [i for i in range(self.n) if membship[big_cluster_id][i] == 1 ]
                rindx = int(self.farest_point[big_cluster_id][0])
                self.MovePoint(rindx, big_cluster_id,ki, representatives_count, representatives_sum,membship,self.X_CATE[rindx],labels_matrix,means_count,means_sum  )
                move +=1

        return move
    def InitClusters(self,representatives,representatives_sum,representatives_count):
        for ki in range(self.k):
            for i in range(self.d):
                sum_ = 0
                for j in range(self.D[i]): sum_ = sum_ + representatives[ki][i][j]
                for j in range(self.D[i]): representatives[ki][i][j] = representatives[ki][i][j]/sum_;

    def DistancePrototypestoAPoints(self,representatives,means, point_CATE,point_NUM):
        dist_matrix_CATE = [self.Distance_CATE(c, point_CATE) for c in representatives]
        dist_matrix_NUM = [self.Distance_NUM(c, point_NUM) for c in means]
        dist_matrix = np.add(np.multiply(dist_matrix_CATE,self.weight_CATE),np.multiply(dist_matrix_NUM,self.weight_NUM))
        representative_id = np.argmin(dist_matrix)
        return representative_id, dist_matrix[representative_id]

    def UpdateLabelsInit(self,representatives, X,representatives_sum, representatives_count,membship,labels_matrix,means_count,means_sum,means):
        cost = 0
        move = 0
        self.preferList= defaultdict(set)
        for ipoint, curpoint in enumerate(X):
            representative_id,tmp = self.DistancePrototypestoAPoints(representatives,means, curpoint, self.X_NUM[ipoint])
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
    def UpdateLabelsLast(self,representatives, X,representatives_sum, representatives_count,membship,labels_matrix,means_count,means_sum,means):
        cost = 0
        self.preferList= defaultdict(set)


        for ipoint, curpoint in enumerate(X):
            representative_id,tmp = self.DistancePrototypestoAPoints(representatives,means, curpoint, self.X_NUM[ipoint])
            cost += tmp
            membship[representative_id, ipoint] = 1
        return cost ,0, 0

    

    def UpdateLabels(self,representatives, X,representatives_sum, representatives_count,membship,labels_matrix,means_count,means_sum,means):
        cost = 0
        move = 0
        for i in range(self.k):
            self.farest_point[i][1] = 0

        for ipoint, curpoint in enumerate(X):
            representative_id,tmp = self.DistancePrototypestoAPoints(representatives,means, curpoint, self.X_NUM[ipoint])
            if tmp > self.farest_point[representative_id][1]:
                self.farest_point[representative_id][1] = tmp
                self.farest_point[representative_id][0] = ipoint
            cost += tmp
            
            if membship[representative_id, ipoint]: continue
            old_clust = labels_matrix[ipoint]
            self.MovePoint(ipoint, old_clust,representative_id, representatives_count, representatives_sum,membship,curpoint,labels_matrix,means_count,means_sum  )
            move +=1
        #Check empty clusters
        move  += self.CheckEmptyClusters(representatives, X,representatives_sum, representatives_count,membship,labels_matrix,means_count,means_sum,means)
        return cost ,move, 0

    def UpdateRepresentatives(self,representatives,representatives_sum,representatives_count,means_count,means_sum,means ) :  
        for ki in range(self.k):
            for di in range(self.d_CATE):
                for vj in range(self.D_CATE[di]):
                    representatives[ki][di][vj] =  representatives_count[ki][di][vj]/representatives_sum[ki]
            
            for di in range(self.d_NUM):
                means[ki][di] = means_sum[ki][di]/means_count[ki]
                
        return 0
    def GetLabels(self, membship):
        labels = np.empty(self.n, dtype=np.uint16)
        for ki in range(self.k):
            for i in range(self.n):
                if membship[ki][i]:
                    labels[i] = ki
        return labels
    def CheckNormalizedData(self):
        isNormalized = True
        for d in range(self.d_CATE):
            if isNormalized == False: break
            for dvalue in range(self.D_CATE[d]):
                if dvalue not in self.uniques_CATE[d]:
                    isNormalized = False; break
        #print("\n\n\n ",isNormalized,"\n\n\n\n\n")
        if isNormalized== False:
            self.normalizer = CategoricalDatasetNormalization(self.X_CATE)
            self.X_CATE = self.normalizer.Normalize(self.X_CATE)
        else: self.normalizer = None

    def DoCluster(self,n_group=2):
        n = len(self.X_NUM)
        d_CATE = self.X_CATE.shape[1]
        d_NUM = self.X_NUM.shape[1]

        self.AddVariableToPrint("n_group",n_group)
        self.k = k = n_clusters = self.k
        self.farest_point = np.zeros((self.k,2))

        self.name = "LSHkRepresentatives"
        
        
        
        


        all_labels = []
        all_costs = []
        start_time = timeit.default_timer()
        #self.n_init=1; # Force set self.n_init
        for init_no in range(self.n_init):
            self.random_state = check_random_state(None)
            membship = np.zeros((k, n), dtype=np.uint8)
            labels_matrix = np.empty(n, dtype=np.uint16)
            
            for i in range(n): labels_matrix[i] = 65535
            
            means_count = np.zeros((self.k))
            means_sum = np.zeros((self.k, self.d_NUM))
            means = np.zeros((self.k, self.d_NUM))

            representatives_count = [[[0 for i in range(self.D_CATE[j])] for j in range(d_CATE)]for ki in range(k)]
            representatives_sum = [0 for ki in range(k)]
            last_cost = float('inf')

            representatives = [[[0 for i in range(self.D_CATE[j])] for j in range(d_CATE)] for ki in range(k)]
            

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
                
                for i in self.lsh.hashTable[key]:
                    labels_matrix[i] = ki
                    membship[ki][i]=1
                    representatives_sum[ki]+=1
                    for ii, val in enumerate(self.X_CATE[i]):
                        representatives_count[ki][ii][val]+=1
                    self.lsh_group[i] = ki

                    means_count[ki] += 1
                    for di in range(self.d_NUM):
                        means_sum[ki][di] += self.X_NUM[i][di]



            self.CheckEmptyClusters(representatives, self.X_CATE,representatives_sum, representatives_count,membship,labels_matrix,means_count,means_count,means)
            self.UpdateRepresentatives(representatives,representatives_sum,representatives_count,means_count,means_sum,means) 
            for i in range(self.n_iter):
                self.iter = i
                cost , move, count_empty = self.UpdateLabels(representatives, self.X_CATE,representatives_sum, representatives_count,membship,labels_matrix,means_count,means_sum,means)
                self.UpdateRepresentatives(representatives,representatives_sum,representatives_count,means_count,means_sum,means ) 
                if last_cost == cost and move==0: 
                    last_cost = self.UpdateLabelsLast(representatives, self.X_CATE,representatives_sum, representatives_count,membship,labels_matrix,means_count,means_sum,means)
                    break 
                last_cost = cost

            labels = self.GetLabels(membship)
            all_costs.append(cost)
            all_labels.append(labels)
            
        best = np.argmin(all_costs)
        labels = all_labels[best]
        self.time_score = (timeit.default_timer() - start_time)/ self.n_init
        self.labels = labels
        print("LSH time:", self.time_lsh ,"Score: ", all_costs[best] , " Time:", self.time_score)
        self.representatives= representatives
        self.means = means
        return self.labels
        
    

    def predict(self,x):    
        if len(x.shape) ==1:
            cateValues = x[self.indexes_CATE]
            if self.normalizer!=None:
                cateValues = self.normalizer.Normalize(cateValues)
            dist_matrix = self.DistancePrototypestoAPoints(self.representatives,self.means,cateValues ,x[self.indexes_NUM].astype(float)  )
            return dist_matrix[0]
        else:
            out = []
            for xi in x:
                out.append(self.predict(xi))
            return np.array(out)

