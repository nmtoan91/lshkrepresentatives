import numpy as np
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
a = kprototypes.fit(X,attributeMasks,numerical_weight=2, categorical_weight=1)
print(a)
print(kprototypes.predict(X[1:5]))


kprototypes2 = LSHkPrototypes(n_clusters=2,n_init=5) 
b = kprototypes2.fit(X)
print(b)
print(kprototypes2.predict(X[1:5]))


X = np.array([[0,0,0,1,1],
              [0,1,1,0,0],
              [0,0,0,3,4],
              [1,5111,1,1.1,1.2],
              [2,2,2,29.0,38.9],
              [2,6513,0,40,41.1],
              [0,0,0,30.4,30.1]])
attributeMasks = [1,1,1,1,1]
kprototypes3 = LSHkPrototypes(n_clusters=2,n_init=5) 
b = kprototypes2.fit(X,attributeMasks)
print(b)
print(kprototypes2.predict(X[1:5]))

