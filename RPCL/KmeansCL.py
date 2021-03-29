from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    dataSet = []
    f = open(fileName)
    for line in f.readlines():
        splitLine = line.strip().split('\t')
        fltLine = list(map(float, splitLine))
        dataSet.append(fltLine)
    return dataSet


def initializeC(dataSet, K):
    col = shape(dataSet)[1]  # acquire number of features
    centroids = mat(zeros((K, col)))
    for i in range(col):
        minCol = min(dataSet[:, i])
        maxCol = max(dataSet[:, i])
        rangeCol = float(maxCol - minCol)
        # create centroids above dataset
        centroids[:, 0] = (minCol + maxCol) / 2 + 2 * random.rand(K, 1) - 1
        centroids[:, 1] = maxCol + 10
    return centroids


def RPCL(dataSet, K, centroids):
    learningRate = 0.2
    gamma = -0.08
    row = shape(dataSet)[0]  # acquire number of points
    cluster = mat(zeros((row, 1)))  # clustering result
    for j in range(row):
        cluster[j] = -1
    distance = mat(zeros((K, 1)))  # distance to each centroid
    a = mat(zeros((K, 1)))  # frequency penalty
    index = random.permutation(row)  # permute sequence
    for k in range(row):
        i = index[k]
        for j in range(K):
            dataNorm = linalg.norm(dataSet[i, :] - centroids[j, :], axis=1, keepdims=True)
            distance[j] = a[j] * (dataNorm ** 2)
        # find winner
        minD = distance[0]
        winner = -1
        for j in range(K):
            if distance[j] <= minD:
                minD = distance[j]
                winner = j
        # find rival
        minD = distance[0] + distance[1]
        rival = 0
        for j in range(K):
            if j == winner:
                continue
            if distance[j] < minD:
                minD = distance[j]
                rival = j
        a[winner] = a[winner] + 1  # add winner frequency
        cluster[i] = winner  # clustering
        # winner move ahead
        centroids[winner, :] = centroids[winner, :] + learningRate * (dataSet[i, :] - centroids[winner, :])
        # rival pull away
        centroids[rival, :] = centroids[rival, :] + learningRate * gamma * (dataSet[i, :] - centroids[rival, :])
    # print(centroids)
    KNum = 0
    for i in range(K):
        if min(dataSet[:, 0]) <= centroids[i, 0] <= max(dataSet[:, 0]) \
           and min(dataSet[:, 1]) <= centroids[i, 1] <= max(dataSet[:, 1]):
            KNum += 1
    KCentroids = mat(zeros((KNum, 2)))
    j = 0
    for i in range(K):
        if min(dataSet[:, 0]) <= centroids[i, 0] <= max(dataSet[:, 0]) \
           and min(dataSet[:, 1]) <= centroids[i, 1] <= max(dataSet[:, 1]):
                KCentroids[j] = centroids[i]
                j += 1
    return KNum, KCentroids


def KMeans(dataSet, K, centroids):
    row = shape(dataSet)[0]
    cluster = mat(zeros((row, 2)))  # clustering result and distance
    converge = False  # decide whether converge
    distance = mat(zeros((K, 1)))  # distance to each centroid
    while not converge:
        converge = True
        for i in range(row):
            for j in range(K):
                dataNorm = linalg.norm(dataSet[i, :] - centroids[j, :], axis=1, keepdims=True)
                distance[j] = dataNorm ** 2
            # find winner
            minD = distance[0]
            winner = -1
            for j in range(K):
                if distance[j] <= minD:
                    minD = distance[j]
                    winner = j
            if cluster[i, 0] != winner:
                converge = False
            cluster[i, 0] = winner
            cluster[i, 1] = minD
        # print(centroids)
        for c in range(K):
            clustered = dataSet[nonzero(cluster[:, 0].A == c)[0]]
            if len(clustered) != 0:
                centroids[c, :] = mean(clustered, axis=0)
    return cluster, centroids


def plotCluster(dataSet, K, cluster, centroids):
    fig = plt.figure()
    colors = ['blue', 'green', 'yellow', 'purple', 'orange', 'black', 'brown']
    ax = fig.add_subplot(111)
    for i in range(K):
        oneCluster = dataSet[nonzero(cluster[:, 0].A == i)[0], :]
        colorStyle = colors[i % len(colors)]
        ax.scatter(oneCluster[:, 0].flatten().A[0], oneCluster[:, 1].flatten().A[0],
                   c=colorStyle)
    ax.scatter(centroids[:, 0].flatten().A[0], centroids[:, 1].flatten().A[0], c='red')


K = 5
dataMat = mat(loadDataSet("\data2.txt"))
centroids = initializeC(dataMat, K)
KNum, KCentroids = RPCL(dataMat, K, centroids)
print(KNum)
print(KCentroids)
myCluster, myCentroids = KMeans(dataMat, KNum, KCentroids)
plotCluster(dataMat, KNum, myCluster, myCentroids)

plt.xlim((-12, 0))
plt.ylim((-8, 8))
plt.show()
