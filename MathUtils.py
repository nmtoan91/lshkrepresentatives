import math
import numpy as np
from decimal import *


# Calculate the frequency of all values in a data set
def countFreq(dataset):
    length = len(dataset[0, :])
    freq = []
    for i in range(length):
        unique, counts = np.unique(dataset[:, i], return_counts=True)
        temp = np.array(list(zip(unique, counts)))
        freq.append(temp)
    return freq


# calculate the frequency of a value of an attribute k
def freq(Ak, a):
    unique, counts = np.unique(Ak, return_counts=True)
    for i in range(len(unique)):
        if unique[i] == a:
            return Decimal(counts[i].item())
    return 0


# calculate the joint frequency of 2 values of 2 attributes
def condFreq(Ar, Al, xr, xl):
    length = len(Ar)
    count = 0
    for i in range(length):
        if Ar[i] == xr and Al[i] == xl:
            count += 1
    return Decimal(count)


# calculate frequency probability of value a in attribute Ak
def prob(Ak, a):
    unique, counts = np.unique(Ak, return_counts=True)
    n = len(Ak)
    for i in range(len(unique)):
        if unique[i] == a:
            return Decimal(counts[i].item()) / n
    return 0


# calculate conditional probability of value ak of attribute Ak known al of attribute Al: P(ak|al)
def condProb(Ak, Al, ak, al):
    count = 0
    total_al = 0
    for i in range(len(Al)):
        if Al[i] == al:
            total_al += 1
            if Ak[i] == ak:
                count += 1
    if total_al == 0:
        return 0
    else:
        return Decimal(count) / total_al


# calculate joint probability between 2 values ak, al of 2 attributes Ak, Al
def jointProb(Ak, Al, ak, al):
    count = 0
    total = 0
    for i in range(len(Ak)):
        if Ak[i] == ak and Al[i] == al:
            count += 1
        total += 1
    if total == 0:
        return 0
    else:
        return Decimal(count) / total


# generate domain matrix for k attributes of data set
def generateAllDomain(dataset):
    length = len(dataset[0, :])
    A = []
    for i in range(length):
        # A.append(generateDomain(dataset[:, i]))
        A.append(np.unique(dataset[:, i]))
    return A


# generate domain for attribute A
def generateDomain(A):
    dom = []
    for j in A:
        if j not in dom:
            dom.append(j)
    return dom


# calculate mutual information between attribute Ak and Al
def mutualInformation(Ak, Al):
    mutualInfor = 0
    domAk = np.unique(Ak)
    domAl = np.unique(Al)
    for p in range(len(domAk)):
        for q in range(len(domAl)):
            Pkl = jointProb(Ak, Al, domAk[p], domAl[q])
            Pk = prob(Ak, domAk[p])
            Pl = prob(Al, domAl[q])
            if Pkl != 0:
                mutualInfor += Pkl * (Pkl / (Pk * Pl)).log10()
    return mutualInfor


# calculate joint entropy between attribute Ak and Al
def jointEntropy(Ak, Al):
    jointEntro = 0
    domAk = np.unique(Ak)
    domAl = np.unique(Al)
    for p in range(len(domAk)):
        for q in range(len(domAl)):
            Pkl = jointProb(Ak, Al, domAk[p], domAl[q])
            if Pkl != 0:
                jointEntro += Pkl * Pkl.log10()
    return jointEntro * (-1)


# calculate entropy of attribute A based on DILCA
def entropy(A):
    entro = 0
    domA = np.array(np.unique(A))
    for a in domA:
        pa = prob(A, a)
        entro += pa * (pa.log10() / Decimal(2).log10())
    return entro * (-1)


# calculate conditional entropy of attribute Ak given Al based on DILCA
def condEntropy(Ak, Al):
    condEntro = 0
    domAk = np.array(np.unique(Ak))
    domAl = np.array(np.unique(Al))
    for i in range(len(domAl)):
        Pli = prob(Al, domAl[i])
        temp = 0
        for j in range(len(domAk)):
            Pkj_li = condProb(Ak, Al, domAk[j], domAl[i])
            if Pkj_li != 0:
                temp += Pkj_li * (Pkj_li.log10() / Decimal(2).log10())
        condEntro += Pli * temp
    return condEntro * (-1)


# calculate information gain of attribute Ak provided by attribute Al based on DILCA
def inforGain(Ak, Al):
    entro = entropy(Ak)
    conEntro = condEntropy(Ak, Al)
    return entro - conEntro


# calculate Symmetric Uncertainty based on DILCA
def symUncertainty(Ak, Al):
    entro_Ak = entropy(Ak)
    entro_Al = entropy(Al)
    inforG = inforGain(Ak, Al)
    if entro_Al + entro_Ak == 0:
        return 0
    else:
        return (2 * inforG) / (entro_Ak + entro_Al)


# calculate correlation between attributes based on ConDist of Ring
def cor(Ak, Al):
    return inforGain(Ak, Al) / entropy(Ak)


# calculate the interdependence redundancy matrix between attributes of data set based on Jia
def interRedundMatrix(dataset):
    length = len(dataset[0, :])
    # RMatrix = np.array([[0 for x in range(length)] for y in range(length)])
    # RMatrix = np.array([0 for x in range(length)])
    RMatrix = []
    for i in range(length):
        # RMatrix[i, i] = 1
        row = []
        for j in range(length):
            interR_ij = interRedund(dataset[:, i], dataset[:, j])
            row.append(interR_ij)
            # RMatrix[i, j] = interR_ij
            # RMatrix[j, i] = RMatrix[i, j]
        RMatrix.append(row)
    return np.array(RMatrix)


# calculate the average dependence of a data set
def avgDependence(dataset):
    RMatrix = interRedundMatrix(dataset)
    k = len(dataset[0, :])
    depend = 0
    count = 0
    for i in range(0, k - 1):
        for j in range(i + 1, k):
            depend += RMatrix[i][j]
            count += 1
    return Decimal(depend) / count


# calculate the symmetric uncertainty metric between attributes of data set based on DILCA
def symUncertaintyMatrix(dataset):
    length = len(dataset[0, :])
    SMatrix = np.array([[0 for x in range(length)] for y in range(length)])
    for i in range(length):
        SMatrix[i][i] = 1
        for j in range(i + 1, length):
            SMatrix[i][j] = symUncertainty(dataset[:, i], dataset[:, j])
            SMatrix[j][i] = symUncertainty(dataset[:, j], dataset[:, i])
    return SMatrix


# caculate mean of symmetric uncertainty for attribute Ak based on DILCA
def meanSymUncertainty(SMatrix, k):
    mean = 0
    length = len(SMatrix[0, :])
    for i in range(length):
        if i != k:
            mean += SMatrix[k][i]
    return float(mean) / (length - 1)


# calculate the interdependence redundancy between 2 attributes based on Jia
def interRedund(Ak, Al):
    I = mutualInformation(Ak, Al)
    H = jointEntropy(Ak, Al)
    if H == 0:
        R = I
    else:
        R = I / H
    return R

def ProconditionMatrixYX( Y, X,Y_unique, X_unique):
        lenData = len(Y)
        lenX = len(X_unique)
        lenY = len(Y_unique)
        HY = 0
        count_x = 0;
        count_y=0;
        MATRIX = [[0 for i in range(lenX)] for j in  range(lenY)]
        for x in range(lenX):
            for y in range(lenY):
                count_x = count_y =0
                for i in range(lenData):
                    if X[i] == x:
                        count_x = count_x+1
                        if(Y[i] == y):
                            count_y = count_y+1;
                MATRIX[y][x] = count_y/count_x if count_x > 0 else 0
        return MATRIX
