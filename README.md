Python implementations of the LSH-k-Representatives algorithms for clustering categorical data:

Different from k-Modes algorithm, LSH-k-Representatives define the "representatives" that keep the frequencies of all categorical values of the clusters.

## Installation:
### Using pip: 
```shell
pip install lshkrepresentatives
```

### Import the packages:
```python
import numpy as np
from LSHkRepresentatives.LSHkRepresentatives_Init import LSHkRepresentatives_Init
from LSHkRepresentatives.LSHkRepresentatives_Full import LSHkRepresentatives_Full
```
### Generate a simple categorical dataset:

```python
X = np.array([[0,0,0],[0,1,1],[0,0,0],[1,0,1],[2,2,2],[2,3,2],[2,3,2]])
y = np.array([0,0,0,0,1,1,1])
```

### LSHk-Representatives (Init): 

```python
kreps = LSHkRepresentatives_Init(X,y,n_init=5,n_clusters=2 ,verbose=3) #variable y is optional for computing the accuracy score, can be replaced as 'None'. 
kreps.SetupLSH()
kreps.DoCluster()

```

### Built-in evaluattion metrics:
```python
kreps.CalcScore()
```

### Outcome:
```shell
Generating disMatrix for DILCA
Saving DILCA to: saved_dist_matrices/json/DILCA_None.json
Generating LSH hash table:   hbits: 2(4)  k 2  d 3  n= 7
LSH time: 0.016015699999999633 Score:  6.333333333333334  Time: 0.0019595600000000602
Purity: 1.00 NMI: 1.00 ARI: 1.00 Sil:  0.59 Acc: 1.00 Recall: 1.00 Precision: 1.00
```



### LSHk-Representatives (Full): 

```python
kreps = LSHkRepresentatives_Full(X,y,n_init=5,n_clusters=2 ,verbose=3) #variable y is optional for computing the accuracy score, can be replaced as 'None'.
kreps.SetupLSH()
kreps.DoCluster()

```

### Built-in evaluattion metrics:
```python
kreps.CalcScore()
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

## Github repository
https://github.com/nmtoan91/lshkrepresentatives