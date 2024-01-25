import numpy as np
from LSHkRepresentatives.LSHkPrototypes import LSHkPrototypes


kprototypes = LSHkPrototypes(n_clusters=2,n_init=5) 
X = np.array([['red',0,np.nan,1,1],['green',1,1,0,0],['blue',0,0,3,4],[1,5111,1,1.1,1.2],[2,2,2,9.0,8.9],[2,6513,'rectangle',4,1.1],[2,3,6565,3.4,3.1]])

attributeMasks = [0,0,0,0,0,0]
a = kprototypes.fit(X,attributeMasks)
print(a)


kprototypes2 = LSHkPrototypes(n_clusters=2,n_init=5) 
b = kprototypes2.fit(X)
print(b)
