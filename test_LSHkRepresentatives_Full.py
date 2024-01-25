import numpy as np
from LSHkRepresentatives.LSHkRepresentatives_Full import LSHkRepresentatives_Full

kreps = LSHkRepresentatives_Full(n_clusters=2,n_init=5) 
X = np.array([['red',0,np.nan],['green',1,1],['blue',0,0],[1,5111,1],[2,2,2],[2,6513,'rectangle'],[2,3,6565]])

labels = kreps.fit(X)

print('Labels:',labels)

x = np.array([1,2,0])
label = kreps.predict(x)