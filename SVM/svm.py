from sklearn import svm
import numpy as np
import pandas as pd
import time

df = pd.read_csv('w/w7a.csv', header=None)  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
X_train = df.iloc[:, 1:]  # 所有行，除第一列以外所有列
X_train = np.array(X_train)
# print(X_train)
y_train = df.iloc[:, 0]
y_train = np.array(y_train)
# print(y_train)
df = pd.read_csv('w/w7at.csv', header=None)
X_test = df.iloc[:, 1:]
X_test = np.array(X_test)
# print(X_test)
y_test = df.iloc[:, 0]
y_test = np.array(y_test)
# print(y_test)

start = time.clock()
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)  # fit svm
end = time.clock()
y_predict = clf.predict(X_test)

correct = 0
y_accuracy = y_predict - y_test
# print(y_accuracy)
for i in y_accuracy:
    if i == 0:
        correct += 1
print("correct=", correct)
print("size=", y_accuracy.shape[0])
print("accuracy=", correct/y_accuracy.shape[0])
print("time=", end-start)


