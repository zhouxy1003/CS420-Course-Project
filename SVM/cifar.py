from six.moves import cPickle as pickle
import numpy as np
import os
from sklearn import svm
import time


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')  # dict类型
        X = datadict['data']  # X, ndarray, 像素值
        Y = datadict['labels']  # Y, list, 标签, 分类

        # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
        # transpose，转置
        # astype，复制，同时指定类型
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []  # list
    ys = []

    # 训练集batch 1～5
    for b in range(1, 5):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)  # 在list尾部添加对象X, x = [..., [X]]
        ys.append(Y)
    Xtr = np.concatenate(xs)  # [ndarray, ndarray] 合并为一个ndarray
    Ytr = np.concatenate(ys)
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    del X, Y

    # 测试集
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)
    return Xtr_rows, Ytr, Xte_rows, Yte


Xtr, Ytr, Xte, Yte = load_CIFAR10('E:\Project\SVM\cifar-10-batches-py')
# print(Xtr.shape[0])
# print(Xtr.shape[1])
# print(Ytr.shape[0])
# print(Xte.shape[0])
# print(Xte.shape[1])
# print(Yte.shape[0])
print('here1')
start = time.clock()
clf = svm.SVC(kernel='rbf')
clf.fit(Xtr, Ytr)  # fit svm
end = time.clock()
print('here2')
Ypd = clf.predict(Xte)
print('here3')

correct = 0
# for i, j in zip(Ypd, Yte):
#     if i == j:
#         print('i=', i)
#         print('j=', j)
#         correct += 1
y_accuracy = Ypd - Yte
for i in y_accuracy:
    if i == 0:
        correct += 1
print("correct=", correct)
print("size=", Ypd.shape[0])
print("accuracy=", correct/Ypd.shape[0])
print("time=", end-start)
