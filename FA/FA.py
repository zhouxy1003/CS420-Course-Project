import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FactorAnalysis


def aic(model, n, m, X):
    dm = n * m + 1 - m * (m - 1) / 2
    return - 2 * model.score(X) + 2 * dm / X.shape[0]


def bic(model, n, m, X):
    dm = n * m + 1 - m * (m - 1) / 2
    return -2 * model.score(X) + np.log(X.shape[0]) * dm / X.shape[0]


N_range = [100, 250, 500, 750, 1000, 5000, 10000]
n_range = [6, 8, 10, 12, 14, 16]
m_range = [2, 3, 4, 5]
sigma_range = [0.1, 0.2, 0.3, 0.4, 0.5]
times = 100
aicAccuracy = []
bicAccuracy = []

for i in range(len(sigma_range)):
    aicCorrect = 0
    bicCorrect = 0
    # generate data
    N = N_range[4]
    n = n_range[2]
    m = m_range[1]
    sigma = sigma_range[i]
    print("N =", N)
    print("n =", n)
    print("m =", m)
    print("sigma =", sigma)
    for time in range(times):
        meanY = np.zeros(m)
        covY = np.identity(m)
        y = np.random.multivariate_normal(meanY, covY, size=N)
        A = np.random.random((m, n))
        meanE = np.zeros(n)
        covE = np.diag(np.ones(n)*sigma)
        e = np.random.multivariate_normal(meanE, covE, size=N)
        X = np.dot(y, A) + e
        # fit the model
        n_components = np.arange(1, n)
        AIC = []
        BIC = []
        models = [FactorAnalysis(n_components=j).fit(X) for j in n_components]
        j = 1
        for model in models:
            AIC.append(aic(model, n, j, X))
            BIC.append(bic(model, n, j, X))
            j += 1
        aicM = AIC.index(min(AIC)) + 1
        bicM = BIC.index(min(BIC)) + 1
        if aicM == m:
            aicCorrect += 1
        if bicM == m:
            bicCorrect += 1
    aicAccuracy.append(aicCorrect / times)
    bicAccuracy.append(bicCorrect / times)
    print("AIC accuracy", aicCorrect / times)
    print("BIC accuracy", bicCorrect / times)

print("AIC accuracy", aicAccuracy)
print("BIC accuracy", bicAccuracy)
