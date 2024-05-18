import numpy as np
from LSHkRepresentatives.LSHkRepresentatives import LSHkRepresentatives
from LSHkRepresentatives.LSHkPrototypes import LSHkPrototypes

X = np.array([
    [0, 0, 0, 0, 0, 0, 1],
    [2, 1, 3, 0, 1, 0, 0],
    [3, 2, 3, 0, 0, 0, 0],
    [3, 3, 3, 0, 0, 1, 1],
    [0, 3, 3, 1, 1, 0, 1],
    [1, 1, 2, 1, 1, 1, 0],
    [1, 1, 2, 1, 1, 1, 0],
    [1, 1, 2, 1, 1, 1, 0],
    [1, 1, 2, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0]
])

attributeMasks = [1, 1, 1, 0, 0, 0, 0]
n_clusters = 4
kprototypes = LSHkPrototypes(n_clusters=n_clusters,n_init=5) 
cluster_labels = kprototypes.fit(X,attributeMasks,numerical_weight=1, categorical_weight=1)