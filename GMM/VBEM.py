import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import BayesianGaussianMixture

N = [100, 250, 500, 750, 1000, 5000, 10000]
K = [2, 3, 4, 5, 6, 7]
D = [2, 3, 4, 5]
SE = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
times = 100
global labels

vbemCorrect = 0
vbemAccuracy = []

for i in range(len(SE)):
    vbemCorrect = 0
    print("N =", N[4])
    print("K =", K[2])
    print("D =", D[1])
    print("SE =", SE[i])
    for time in range(times):
        data, tag = make_blobs(n_samples=N[4], centers=K[2], n_features=D[1], cluster_std=SE[i], center_box=(-20.0, 20.0))
        mixture = BayesianGaussianMixture(n_components=10, covariance_type='full', max_iter=200,
                                          weight_concentration_prior_type='dirichlet_distribution')
        gmm = mixture.fit(data)
        labels = gmm.predict(data)
        KSet = set(labels)
        vbemK = len(KSet)
        if vbemK == K[2]:
            vbemCorrect += 1
    vbemAccuracy.append(vbemCorrect / times)

print("VBEM accuracy", vbemAccuracy)
plt.plot(SE, vbemAccuracy)
plt.xlabel('SE')
plt.ylabel('Accuracy')
plt.show()
