import os
import os.path
import sys
from sys import platform
sys.path.append(os.path.join(os.getcwd(), "Measures"))
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
import timeit

import random
from .SimpleHashing import SimpleHashing
from collections import defaultdict
import statistics 
from collections import Counter
class LSH(SimpleHashing):
    def DoHash(self):
        self.measure.GeneratesimMatrix()
        self.GenerateSimilarityMatrix(self.measure.simMatrix)
        self.bit_indexes  = np.argpartition(self.cut_values_normal, self.hbits)[:self.hbits]   
        self.GenerateHashTable()
        return -1

    def GenerateHashTable(self):
        print("Generating LSH hash table: ", " hbits:", str(self.hbits) +'('+ str(2**self.hbits)+')', " k", self.k , " d", self.d , " n=",self.n )
        self.hash_values = [self.ComputeHashValue(x) for x in self.X]
        self.hashTable = defaultdict(list)
        for i in range(self.n):
            self.hashTable[self.hash_values[i]].append(i)

    def GetNeighborsbyBucket(self, item_id):
        return self.hashTable[self.hash_values[item_id]]

    def ComputeHashValue(self,x):
        val=0
        for i in range(self.hbits):
            partitions = self.partitions[self.bit_indexes[i]]
            val <<=1
            if x[self.bit_indexes[i]] in partitions[1]:
                val+=1
        return val
    #def ComputeHashValue(self,x):
    #    val=0
    #    for i in range(self.hbits):
    #        #partitions = self.partitions[self.bit_indexes[i]]
    #        val <<=1
    #        #if x[self.bit_indexes[i]] in partitions[1]:
    #        if x[self.bit_indexes[i]] >= self.D[self.bit_indexes[i]]/2:
    #            val+=1
    #    return val
    #def TryToMovePoint(self,hashValue_ori, hashValue_cur, index ):
        #for i in range(index,self.hbits):

    def hammingDistance(self, x, y):
        ans = 0
        for i in range(31,-1,-1):
            b1= x>>i&1
            b2 = y>>i&1
            ans+= not(b1==b2)
        return ans
    def CorrectSingletonBucket(self):
        list_hash_1 = []
        for hashValue,itemList in self.hashTable.items():
            if(len(itemList)<=1): 
                list_hash_1.append((hashValue, itemList))

        for iters in list_hash_1:
            del self.hashTable[iters[0]]

        for iters in list_hash_1:
            closest_hash_value = -1
            closest_dist=1000000
            for hashValue,itemList in self.hashTable.items():
                temp = self.hammingDistance(iters[0],hashValue)
                if temp < closest_dist:
                    closest_hash_value = hashValue
                    closest_dist = temp
            for i in iters[1]:
                self.hash_values[i] =  closest_hash_value
                self.hashTable[closest_hash_value].append(i)
        print("LSH Merged ",len(list_hash_1),"/", len(self.hashTable) , " buckets!!!" )

    def TestHashTable(self):
        n = len(self.hashTable.items())
        num_0 = 2**self.hbits - len(self.hashTable.items());
        num_1 = 0;
        len_list=[]

        for hashValue,itemList in self.hashTable.items():
            if(len(itemList)==1): num_1+=1
            len_list.append(len(itemList))
        mean=np.mean(len_list)
        std_ = np.std(len_list)

        print( "Num bucket:",n," Num zero:", num_0, " Num 1:", num_1, " Mean:", mean, " Std:",std_)
        #Test within buckets
        sum_=0
        for hashValue,itemList in self.hashTable.items():
            labels = self.y[itemList]
            test_list = Counter(labels) 
            domiant_label = test_list.most_common(1)[0][0] 
            sum_+= sum(labels==domiant_label)
        return sum_/self.n
def main():
    MeasureManager.CURRENT_MEASURE = 'DILCA'
    DB = tulti.LoadSynthesisData(n=2048,d=10,k=20,sigma_rate=0.1);
    MeasureManager.CURRENT_DATASET = DB['name']
    hashing = LSH(DB['DB'],DB['labels_'],measure='DILCA')
    hashing.test()
    hashing.DoHash()
    score = hashing.TestHashTable()
    print('Score: ', score)
    hashing.CorrectSingletonBucket();
    score = hashing.TestHashTable()
    print('Score: ', score)
    asd = 123
if __name__ == "__main__":
    main()