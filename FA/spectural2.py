from sklearn.datasets import make_blobs
from itertools import cycle, islice
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import warnings
from sklearn.mixture import GaussianMixture

data, tag = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=[1, 1, 1.5], center_box=(-10, 10))
transformation = [[0.6, -0.6], [-0.4, 0.8]]
data = np.dot(data, transformation)
spectral = cluster.SpectralClustering(
        n_clusters=3, eigen_solver='arpack',
        affinity="nearest_neighbors")
# catch warnings related to kneighbors_graph
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Graph is not fully connected, spectral embedding" +
        " may not work as expected.",
        category=UserWarning)
    spectral.fit(data)
y_pred = spectral.labels_.astype(np.int)
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
# add black color for outliers (if any)
colors = np.append(colors, ["#000000"])
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(data[:, 0], data[:, 1], s=10, color=colors[y_pred])


gauss = GaussianMixture(n_components=3, covariance_type='full').fit(data)
y_pred = gauss.predict(data)
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(data[:, 0], data[:, 1], s=10, color=colors[y_pred])
plt.show()
