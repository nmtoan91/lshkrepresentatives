#Note by toanstt: The short lists of near clusters are updated each iterator

import os
import os.path
import sys
from sys import platform
sys.path.append(os.path.join(os.getcwd(), "Measures"))
sys.path.append(os.path.join(os.getcwd(), "LSH"))
sys.path.append(os.path.join(os.getcwd(), "../"))
sys.path.append(os.path.join(os.getcwd(), "../Dataset"))
sys.path.append(os.path.join(os.getcwd(), "../Measures"))
sys.path.append(os.path.join(os.getcwd(), "../LSH"))
sys.path.append(os.path.join(os.getcwd(), "./ClusteringAlgorithms"))
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


class LSHkRepresentatives_Full(ClusteringAlgorithm):

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
        sum=0;
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
                time_start_found = timeit.default_timer()
                big_cluster_id = np.argmax([sum(mem_) for mem_ in membship])
                choices = [i for i in range(self.n) if membship[big_cluster_id][i] == 1 ]
                rindx = int(self.farest_point[big_cluster_id][0])
                self.MovePoint(rindx, big_cluster_id,ki, representatives_count, representatives_sum,membship,self.X[rindx],labels_matrix  )
                move +=1
                if TDef.verbose >=2 :  print("Found a empty cluster (Loop): ", ki, " Timelapse=", "%.2f"%(timeit.default_timer()-time_start_found))
        return move
    def CheckEmptyClusters_Init(self,representatives, X,representatives_sum, representatives_count,membship,labels_matrix):
        count =0;
        move =0
        big_cluster_id = -1
        is_have = False
        for ki in range(self.k):
            if representatives_sum[ki] ==0 :
                count+=1
                is_have = True
                
                rindx = random.randint(0, self.n-1) 
                big_cluster_id = labels_matrix[rindx]
                self.MovePoint(rindx, big_cluster_id,ki, representatives_count, representatives_sum,membship,self.X[rindx],labels_matrix  )
                move +=1
        #if TDef.verbose >=3:  print("WARING: Found " + str(count) +"/" + str(self.k) + " empty clusters (Init) ")
        if is_have: return move + self.CheckEmptyClusters_Init(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
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
            #representative_id,tmp = self.DistanceRepresentativestoAPoints(representatives, curpoint)
            representative_id,tmp = self.DistanceRepresentativestoAPoints_LSH2(ipoint, curpoint,labels_matrix,representatives)
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
    def UpdateClusterDistMatrix(self,representatives):
        for i in range(self.k):
            for j in range(i+1,self.k):
                sum =0
                for k in range(self.d):
                    sum+= np.linalg.norm(representatives[i][k] - representatives[j][k])
                self.cluster_dist_matrix[i][j] = self.cluster_dist_matrix[j][i] =  sum
            self.near_clusters[i] = np.argsort(self.cluster_dist_matrix[i])[0:self.nearcluster_count+1]
        #print(self.near_clusters[0])
    def DoCluster(self,n_group=2,init_type ='random'):
        if n_group==-1:
            if self.k >= 8: n_group = self.k/4;
            else: n_group=2
        self.AddVariableToPrint("init_type",init_type)
        self.AddVariableToPrint("n_group",n_group)
        self.k = k = n_clusters = self.k
        self.farest_point = np.zeros((self.k,2))

        self.name = "LSHkRepresentatives_Full"
        #print("LSHkRepresentatives start clustering")
        #Init varibles
        X = self.X
        self.n = n = self.X.shape[0];
        self.d = d = X.shape[1]
        self.D = D = [len(np.unique(X[:,i])) for i in range(d) ]
        all_labels = []
        all_costs = []
        start_time = timeit.default_timer()
        
        self.cluster_dist_matrix = np.zeros((self.k,self.k))
        self.near_clusters = [[] for j in range(self.k)]

        print(" n_group=" + str(n_group) + " Average neighbors:" + str(self.k/n_group) )
        for init_no in range(self.n_init):
            start_time_init = timeit.default_timer()
            #if TDef.verbose >=1: print (self.name +' Init ' + str(init_no))
            self.random_state = check_random_state(None)
            membship = np.zeros((k, n), dtype=np.uint8)
            labels_matrix = np.empty(n, dtype=np.uint16)
            
            #if TDef.verbose >=3: print("Initing memories");
            for i in range(n): labels_matrix[i] = 65535
            representatives_count = [[[0 for i in range(D[j])] for j in range(d)]for ki in range(k)]
            representatives_sum = [0 for ki in range(k)]
            last_cost = float('inf')

            #representatives = [[[0 for i in range(D[j])] for j in range(d)] for ki in range(k)]
            representatives = [[np.zeros(D[j]) for j in range(d)] for ki in range(k)]

            buckets = [(k,len(self.lsh.hashTable[k])) for k in self.lsh.hashTable.keys()]
            buckets2 = sorted(buckets, key=lambda x: -x[-1])

            masterkeys=[]
            n_group = int(self.k/n_group)
            self.nearcluster_count=n_group
            for i in range( self.k - len(buckets2)):
                buckets2.append((0,0))
            for i in range(self.k):
                masterkeys.append(buckets2[i][0])
            #if TDef.verbose >=3: print("Computing dist matrices");
            dist_from_master_to_master = np.zeros((self.k,self.k))

            for i in range(self.k):
                for j in range(i+1,self.k):
                    dist_from_master_to_master[i][j]= dist_from_master_to_master[j][i] = \
                    self.lsh.hammingDistance(masterkeys[i],masterkeys[j])

            k2 = len(buckets2)
            for i in range(k2):
                buckets = self.lsh.hashTable[buckets2[i][0]]
                if i < k: i2= i
                else : 
                    if init_type== 'best':
                        i2 = -1; dst= float('inf')
                        for i3 in range(k):
                            dst_tmp = self.lsh.hammingDistance(buckets2[i][0],masterkeys[i3])
                            if dst_tmp<=dst: dst = dst_tmp; i2=i3
                    else: i2 = random.randint(0,self.k-1)
                for j in buckets:
                    representatives_sum[i2]+=1
                    labels_matrix[j] = i2
                    membship[i2][j] = 1
                    for ii, val in enumerate(X[j]):
                        representatives_count[i2][ii][val]+=1

            #if TDef.verbose >=3: print ('Checking Empty Clusters')
            self.CheckEmptyClusters_Init(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
            #if TDef.verbose >=3: print ('UpdateRepresentatives')
            self.UpdateRepresentatives(representatives,representatives_sum,representatives_count ) ;
            self.UpdateClusterDistMatrix(representatives)
            #if TDef.verbose >=1: print (' Init timelapse: ' + str(timeit.default_timer() -start_time_init))
            for i in range(self.n_iter):
                start_time_iter = timeit.default_timer()
                self.iter = i
                cost , move, count_empty = self.UpdateLabels(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
                self.UpdateRepresentatives(representatives,representatives_sum,representatives_count ) ;
                self.UpdateClusterDistMatrix(representatives)
                #if TDef.verbose >=2: print ('Iter ' + str(i),"Cost: ", "%.2f"%cost," Move:", move, " Num empty:", count_empty," Timelapse:","%.2f"% (timeit.default_timer()-start_time_iter ))
                if last_cost == cost and move==0: 
                    last_cost,_,_ = self.UpdateLabelsLast(representatives, X,representatives_sum, representatives_count,membship,labels_matrix)
                    #if TDef.verbose >=2: print ('Iter (last)' ,"Cost: ", "%.2f"%last_cost," Move:", move, " Num empty:", count_empty," Timelapse:","%.2f"% (timeit.default_timer()-start_time_iter ))
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
        self.scorebest = all_costs[best]
        self.representatives = representatives
        return self.labels
        # Update representives
    def predict(self,x):
        dist_matrix = self.DistanceRepresentativestoAPoints(self.representatives, x)
        return dist_matrix[0]
    
def Test(): 
    #MeasureManager.CURRENT_DATASET = 'tae_c.csv'
    MeasureManager.CURRENT_DATASET = 'lung.csv'
    MeasureManager.CURRENT_MEASURE = 'Overlap'
    if TDef.data!='': MeasureManager.CURRENT_DATASET = TDef.data
    if TDef.measure!='': MeasureManager.CURRENT_MEASURE = TDef.measure
    if TDef.test_type == 'syn':
        DB = tulti.LoadSynthesisData(TDef.n,  TDef.d, TDef.k)
        MeasureManager.CURRENT_DATASET= DB['name']
    else:
        DB = tulti.LoadRealData(MeasureManager.CURRENT_DATASET)
    print("\n\n############## LSHkRepresentatives_Full ###################")
    lshkrepresentatives = LSHkRepresentatives_Full(DB['DB'],DB['labels_'] ,dbname=MeasureManager.CURRENT_DATASET ,k=TDef.k)
    lshkrepresentatives.SetupMeasure(MeasureManager.CURRENT_MEASURE)
    lshkrepresentatives.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
    lshkrepresentatives.DoCluster(n_group=TDef.n_group,init_type=TDef.init_type)
    lshkrepresentatives.CalcScore()

def TestDatasets(): 
    for dbname in MeasureManager.DATASET_LIST:
        DB = tulti.LoadRealData(dbname)
        MeasureManager.CURRENT_DATASET = dbname
        MeasureManager.CURRENT_MEASURE = 'Overlap'
        print("\n\n############## "+self.name+" ###################")
        alo = LSHkRepresentatives_Full(DB['DB'],DB['labels_'],dbname=MeasureManager.CURRENT_DATASET )
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
            alo = LSHkRepresentatives_Full(DB['DB'],DB['labels_'],dbname=MeasureManager.CURRENT_DATASET )
            alo.SetupMeasure(MeasureManager.CURRENT_MEASURE)
            alo.SetupLSH(measure=MeasureManager.CURRENT_MEASURE)
            alo.DoCluster()
            alo.CalcScore()

            alo.AddValuesToMyTable(table,measure == MeasureManager.MEASURE_LIST[0],dataset)
            table.SaveToExcelFolder("RESULT/r20200916" + alo.name,TDef.o,[i+1 for i in range(0,1000)])

if __name__ == "__main__":
    TDef.InitParameters(sys.argv)
    if TDef.test_type == 'datasets':
        TestDatasets()
    elif TDef.test_type == 'measures':
        TestMeasures()
    else:
        Test()