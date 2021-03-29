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
    col = shape(dataSet)[1]  # 获取数据集特征值数
    centroids = mat(zeros((K, col)))   # 创建K行col列零矩阵
    for i in range(col):  # 找出数据集每列的最大值
        minCol = min(dataSet[:, i])
        maxCol = max(dataSet[:, i])
        rangeCol = float(maxCol - minCol)
        # 在数据集上方生成质心
        centroids[:, 0] = (minCol + maxCol) / 2 + 2 * random.rand(K, 1) - 1
        centroids[:, 1] = maxCol + 10
        # 在数据集右上角生成质心
        # centroids[:, i] = maxCol + 10 + random.rand(K, 1)
        # 在数据集内部生成质心
        # centroids[:, i] = minCol + rangeCol * random.rand(K, 1)
        # 在数据集中心生成质心
        # centroids[:, i] = (minCol + maxCol) / 2 + random.rand(K, 1) - 0.5
    return centroids


def RPCL(dataSet, K, centroids):
    learningRate = 0.2
    gamma = -0.08
    row = shape(dataSet)[0]  # 获取数据集行数
    cluster = mat(zeros((row, 1)))  # 分类结果
    for j in range(row):
        cluster[j] = -1
    distance = mat(zeros((K, 1)))  # 点到每个质心的距离
    a = mat(zeros((K, 1)))  # 次数惩罚
    index = random.permutation(row)  # 打乱索引顺序
    for k in range(row):
        i = index[k]
        for j in range(K):
            dataNorm = linalg.norm(dataSet[i, :] - centroids[j, :], axis=1, keepdims=True)
            distance[j] = a[j] * (dataNorm ** 2)
        # 寻找winner
        minD = distance[0]
        winner = -1
        for j in range(K):
            if distance[j] <= minD:
                minD = distance[j]
                winner = j
        # 寻找rival
        minD = distance[0] + distance[1]
        rival = 0
        for j in range(K):
            if j == winner:
                continue
            if distance[j] < minD:
                minD = distance[j]
                rival = j
        a[winner] = a[winner] + 1  # winner次数增加1
        cluster[i] = winner  # 分类完成
        # winner前进
        centroids[winner, :] = centroids[winner, :] + learningRate * (dataSet[i, :] - centroids[winner, :])
        # rival推走
        centroids[rival, :] = centroids[rival, :] + learningRate * gamma * (dataSet[i, :] - centroids[rival, :])
    return cluster, centroids


def plotCluster(dataSet, K, cluster, centroids):
    fig = plt.figure()
    colors = ['blue', 'green', 'yellow', 'purple', 'orange', 'black', 'brown']
    ax = fig.add_subplot(111)
    for i in range(K):
        oneCluster = dataSet[nonzero(cluster[:, 0].A == i)[0], :]
        colorStyle = colors[i % len(colors)]
        ax.scatter(oneCluster[:, 0].flatten().A[0], oneCluster[:, 1].flatten().A[0], c=colorStyle)
    ax.scatter(centroids[:, 0].flatten().A[0], centroids[:, 1].flatten().A[0], c='red')


K = 5
dataMat = mat(loadDataSet("\data2.txt"))
# plt.scatter(dataMat[:, 0].tolist(), dataMat[:, 1].tolist())
centroids = initializeC(dataMat, K)
# plt.scatter(centroids[:, 0].tolist(), centroids[:, 1].tolist())
myCluster, myCentroids = RPCL(dataMat, K, centroids)
plotCluster(dataMat, K, myCluster, myCentroids)
# plt.scatter(myCentroids[:, 0].tolist(), myCentroids[:, 1].tolist())

plt.xlim((-12, 0))
plt.ylim((-8, 8))

# plt.xlim((-10, 16))
# plt.ylim((-14, 0))
plt.show()


