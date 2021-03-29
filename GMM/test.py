import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# K = 6
# D = 2
# data, tag = make_blobs(n_samples=10000, centers=K, n_features=D, cluster_std=1.5, center_box=(-20.0, 20.0))
# gmm = GaussianMixture(n_components=K).fit(data)
# labels = gmm.predict(data)
# plt.scatter(data[:, 0], data[:, 1], c=labels, s=10, cmap='viridis')
# plt.show()

N = [100, 250, 500, 750, 1000, 5000, 10000]
K = [2, 3, 4, 5, 6, 7]
D = [2, 3, 4, 5]
SE = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

aicAccuracy = [0.96, 0.97, 0.95, 0.97, 0.96, 0.95, 0.94]
bicAccuracy = [1.0, 0.99, 0.99, 0.98, 0.98, 0.95, 0.93]
vbemAccuracy = [1.0, 0.98, 0.98, 0.98, 0.96, 0.97, 0.95]

plt.plot(SE, aicAccuracy, label='AIC')
plt.plot(SE, bicAccuracy, label='BIC')
plt.plot(SE, vbemAccuracy, label='VBEM')
plt.legend(loc='best')
plt.xlabel('SE')
plt.ylabel('Accuracy')
plt.xticks(SE)
plt.xlim(0.5, 2.0)
# plt.ylim(0.92, 1.01)
plt.show()
