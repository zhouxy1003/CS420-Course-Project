import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

N = [100, 250, 500, 750, 1000, 5000, 10000]
K = [2, 3, 4, 5, 6, 7]
D = [2, 3, 4, 5]
SE = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
times = 100
aicAccuracy = []
bicAccuracy = []

for i in range(len(SE)):
    aicCorrect = 0
    bicCorrect = 0
    print("N =", N[4])
    print("K =", K[2])
    print("D =", D[1])
    print("SE =", SE[i])
    for time in range(times):
        # generate GMM data
        data, tag = make_blobs(n_samples=N[4], centers=K[2], n_features=D[1], cluster_std=SE[i], center_box=(-20.0, 20.0))
        n_components = np.arange(1, 10)
        models = [GaussianMixture(n, covariance_type='full').fit(data) for n in n_components]
        AIC = [m.aic(data) for m in models]
        BIC = [m.bic(data) for m in models]
        aicK = AIC.index(min(AIC)) + 1
        bicK = BIC.index(min(BIC)) + 1
        # print(aicK)
        # print(bicK)
        if aicK == K[2]:
            aicCorrect += 1
        if bicK == K[2]:
            bicCorrect += 1
    aicAccuracy.append(aicCorrect / times)
    bicAccuracy.append(bicCorrect / times)

print("AIC accuracy", aicAccuracy)
print("BIC accuracy", bicAccuracy)


plt.plot(SE, aicAccuracy, label='AIC')
plt.plot(SE, bicAccuracy, label='BIC')
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Accuracy')
# plt.xticks(SE)
# plt.xlim(0.5, 2.0)
# plt.ylim(0.92, 1.01)
plt.show()
