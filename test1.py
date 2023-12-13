import numpy as np
from LSHkRepresentatives.LSHkRepresentatives import LSHkRepresentatives

X = np.array([[0,0,0],[0,1,1],[0,0,0],[1,0,1],[2,2,2],[2,3,2],[2,3,2]])
kreps = LSHkRepresentatives(n_clusters=2,n_init=5) 
labels = kreps.fit(X)

print('Labels:',labels)

x = np.array([1,2,0])

label = kreps.predict(x)

print(f'Cluster of object {x} is: {label}')

x2 = np.array([[1,2,0],[2,2,2]])

label2 = kreps.predict(x2)

print(label2)

y = np.array([0,0,0,0,1,1,1])
kreps.CalcScore(y)