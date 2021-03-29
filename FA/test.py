import matplotlib.pyplot as plt

N_range = [100, 250, 500, 750, 1000, 5000, 10000]
n_range = [6, 8, 10, 12, 14, 16]
m_range = [2, 3, 4, 5]
sigma_range = [0.1, 0.2, 0.3, 0.4, 0.5]


aicAccuracy = [0.77, 0.81, 0.84, 0.83, 0.88]
bicAccuracy = [1.0, 0.99, 0.95, 0.97, 0.87]

plt.plot(sigma_range, aicAccuracy, label='AIC')
plt.plot(sigma_range, bicAccuracy, label='BIC')
plt.legend(loc='best')
plt.xlabel('sigma^2')
plt.ylabel('Accuracy')
plt.xticks(sigma_range)
# plt.xlim(0.5, 2.0)
# plt.ylim(0.92, 1.01)
plt.show()
