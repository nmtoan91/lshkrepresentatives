# Clustering algorithm for Mixed data of categorial and numerical (ordinal and nonordinal) data using LSH.
## Notebook samples:
### 1. LSH-k-Representatives : Clustering of categorical attributes only:
### https://github.com/nmtoan91/lshkrepresentatives/blob/main/notebook_sample_clustering_categorical_data.ipynb

### 2. LSH-k-Prototypes : Clustering of mixed data (categorical and numerical attributes):
### https://github.com/nmtoan91/lshkrepresentatives/blob/main/notebook_sample_clustering_mixed_data_type.ipynb

### 3. LSH-k-Representatives-Full : Clustering of HUGE categorical attributes only:
### https://github.com/nmtoan91/lshkrepresentatives/blob/main/notebook_sample_LSHkRepresentatives_Full.ipynb

### 4. Normalizing unstructed normal dataset: 
### https://github.com/nmtoan91/lshkrepresentatives/blob/main/notebook_dataset_normalization.ipynb

<br />
<br />
Note 1: Different from k-Modes algorithm, LSH-k-Representatives define the "representatives" that keep the frequencies of all categorical values of the clusters. There are threee algorithms 
Note 2: The dataset is auto normalized if it detect string, or disjointed data, or nan 


## Installation:
### Using pip: 
```shell
pip install lshkrepresentatives numpy scikit-learn pandas networkx termcolor
```

### Import the packages:
```python
import numpy as np
from LSHkRepresentatives.LSHkRepresentatives import LSHkRepresentatives
```
### Generate a simple categorical dataset:

```python
X = np.array([['red',0,np.nan],['green',1,1],['blue',0,0],[1,5111,1],[2,2,2],[2,6513,'rectangle'],[2,3,6565]])
```

## Using LSHk-Representatives (categorical clustering): 

```python
#Init instance of LSHkRepresentatives 
kreps = LSHkRepresentatives(n_clusters=2,n_init=5) 
#Do clustering for dataset X
labels = kreps.fit(X)
#Print the label for dataset X
print('Labels:',labels)
#Predict label for the random instance x
x = np.array(['red',5111,0])
label = kreps.predict(x)
print(f'Cluster of object {x} is: {label}')
```

#### Outcome:
```shell
SKIP LOADING distMatrix because: False bd=None
Generating disMatrix for DILCA
Saving DILCA to: saved_dist_matrices/json/DILCA_None.json
Generating LSH hash table:   hbits: 2(4)  k 1  d 3  n= 7
LSH time: 0.006518099999993865 Score:  6.333333333333334  Time: 0.0003226400000130525
Labels: [1 1 1 1 0 0 0]
Cluster of object [1 2 0] is: 1
```

### Call built-in evaluattion metrics:
```python
y = np.array([0,0,0,0,1,1,1])
kreps.CalcScore(y)
```
#### Outcome:
```shell
Purity: 1.00 NMI: 1.00 ARI: 1.00 Sil:  0.59 Acc: 1.00 Recall: 1.00 Precision: 1.00
```

## Using LSHk-Prototypes (Mixed categorical and numerical attributes clustering): 
For example: We have a dataset of 5 attributes (3 categorical and 2 numerical).
```python
from LSHkRepresentatives.LSHkPrototypes import LSHkPrototypes
kprototypes = LSHkPrototypes(n_clusters=2,n_init=5) 
X = np.array([['red',0,np.nan,1,1],
              ['green',1,1,0,0],
              ['blue',0,0,3,4],
              [1,5111,1,1.1,1.2],
              [2,2,2,29.0,38.9],
              [2,6513,'rectangle',40,41.1],
              ['red',0,np.nan,30.4,30.1]])

attributeMasks = [0,0,0,1,1]
# attributeMasks = [0,0,0,1,1] means attributes are
# [categorial,categorial,categorial,numerical,numerical]
a = kprototypes.fit(X,attributeMasks,numerical_weight=2, categorical_weight=1)
print(a)
```


## References:
T. N. Mau and V.-N. Huynh, ``An LSH-based k-Representatives Clustering Method for Large Categorical Data." Neurocomputing,
			Volume 463, 2021, Pages 29-44, ISSN 0925-2312, https://doi.org/10.1016/j.neucom.2021.08.050.

## Bibtex:
```
@article{mau2021lsh,
  title={An LSH-based k-representatives clustering method for large categorical data},
  author={Mau, Toan Nguyen and Huynh, Van-Nam},
  journal={Neurocomputing},
  volume={463},
  pages={29--44},
  year={2021},
  publisher={Elsevier}
}
```
## pypi/github repository
https://pypi.org/project/lshkrepresentatives/ \
https://github.com/nmtoan91/lshkrepresentatives
