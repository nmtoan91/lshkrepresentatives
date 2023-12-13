Python implementations of the LSH-k-Representatives algorithms for clustering categorical data:

Different from k-Modes algorithm, LSH-k-Representatives define the "representatives" that keep the frequencies of all categorical values of the clusters.

## Notebook sample: 
https://github.com/nmtoan91/lshkrepresentatives/blob/main/LSHkRepresentatives_notebook_sample.ipynb

## Installation:
### Using pip: 
```shell
pip install lshkrepresentatives numpy scikit-learn pandas kmodes networkx termcolor

```

### Import the packages:
```python
import numpy as np
from LSHkRepresentatives.LSHkRepresentatives import LSHkRepresentatives
```
### Generate a simple categorical dataset:

```python
X = np.array([[0,0,0],[0,1,1],[0,0,0],[1,0,1],[2,2,2],[2,3,2],[2,3,2]])
```

### LSHk-Representatives (Init): 

```python
#Init instance of LSHkRepresentatives 
kreps = LSHkRepresentatives(n_clusters=2,n_init=5) 
#Do clustering for dataset X
labels = kreps.fit(X)
#Print the label for dataset X
print('Labels:',labels)
#Predict label for the random instance x
x = np.array([1,2,0])
label = kreps.predict(x)
print(f'Cluster of object {x} is: {label}')
```

### Outcome:
```shell
SKIP LOADING distMatrix because: False bd=None
Generating disMatrix for DILCA
Saving DILCA to: saved_dist_matrices/json/DILCA_None.json
Generating LSH hash table:   hbits: 2(4)  k 1  d 3  n= 7
LSH time: 0.006518099999993865 Score:  6.333333333333334  Time: 0.0003226400000130525
Labels: [1 1 1 1 0 0 0]
Cluster of object [1 2 0] is: 1
```

### Built-in evaluattion metrics:
```python
y = np.array([0,0,0,0,1,1,1])
kreps.CalcScore(y)
```
### Outcome:
```shell
Purity: 1.00 NMI: 1.00 ARI: 1.00 Sil:  0.59 Acc: 1.00 Recall: 1.00 Precision: 1.00
```


### LSHk-Representatives (Full): 
This version of LSHk-Representatives target for huge dataset, the accuracy will be reduced but the speed is increase from 2 to 32 times depend on the data

```python
X = np.array([[0,0,0],[0,1,1],[0,0,0],[1,0,1],[2,2,2],[2,3,2],[2,3,2]])
kreps = LSHkRepresentatives_Full(n_clusters=2,n_init=5) 
labels = kreps.fit(X)
print('Labels:',labels)
x = np.array([1,2,0])
label = kreps.predict(x)
print(f'Cluster of object {x} is: {label}')
```

### Built-in evaluattion metrics:
```python
y = np.array([0,0,0,0,1,1,1])
kreps.CalcScore(y)
```

### Out come:
```shell
SKIP LOADING distMatrix because: True bd=None
Generating disMatrix for DILCA
Saving DILCA to: saved_dist_matrices/json/DILCA_None.json
Generating LSH hash table:   hbits: 2(4)  k 2  d 3  n= 7
 n_group=2 Average neighbors:1.0
LSH time: 0.00661619999999985 Score:  6.333333333333334  Time: 0.000932080000000024
Purity: 1.00 NMI: 1.00 ARI: 1.00 Sil:  0.59 Acc: 1.00 Recall: 1.00 Precision: 1.00
```

## Parameters:
X: Categorical dataset\
y: Labels of object (for evaluation only)\
n_init: Number of initializations \
n_clusters: Number of target clusters\
max_iter: Maximum iterations\
verbose: \
random_state: 

If the variable MeasureManager.IS_LOAD_AUTO is set to "True": The DILCA will get the pre-caculated matrix
 
## Outputs:
cluster_representatives: List of final representatives\
labels_: Prediction labels\
cost_: Final sum of squared distance from objects to their centroids\
n_iter_: Number of iterations\
epoch_costs_: Average time for an initialization


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
