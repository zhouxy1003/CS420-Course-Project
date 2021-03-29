import matplotlib.pyplot as plt

x_label = [5, 7, 10, 15, 20, 33, 50]
a_size = [4.93, 6.96, 9.78, 14.68, 19.70, 34.46, 49.45]
w_size = [4.98, 6.98, 9.87, 14.81, 19.88, 34.55, 49.63]
svm_rbf = [0.8421, 0.8426, 0.8428, 0.8447, 0.845, 0.8462, 0.8466]
svm_linear = [0.8382, 0.8428, 0.8433, 0.8445, 0.8438, 0.8473, 0.8484]
svm_poly = [0.8402, 0.8399, 0.8394, 0.8427, 0.8437, 0.847, 0.8472]
mlp_lbfgs = [0.802, 0.7958, 0.7937, 0.8012, 0.8, 0.8062, 0.8053]
mlp_adam = [0.8295, 0.8128, 0.8127, 0.8191, 0.8187, 0.8153, 0.8242]
svm_rbf2 = [0.9738, 0.9765, 0.9771, 0.9804, 0.9806, 0.9852, 0.9859]
svm_linear2 = [0.9774, 0.9807, 0.9829, 0.984, 0.985, 0.9865, 0.9868]
svm_poly2 = [0.9724, 0.9726, 0.9729, 0.9735, 0.9741, 0.9779, 0.9799]
mlp_lbfgs2 = [0.9714, 0.9755, 0.9786, 0.9821, 0.9841, 0.985, 0.9869]
mlp_adam2 = [0.9787, 0.9806, 0.9829, 0.9841, 0.9847, 0.9869, 0.9884]

method = ['SVM RBF', 'SVM Linear', 'SVM Poly', 'MLP lbfgs', 'MLP adam']
a_accuracy = [0.8466, 0.8484, 0.8472, 0.8053, 0.8242]
w_accuracy = [0.9859, 0.9868, 0.9799, 0.9869, 0.9884]
g_accuracy = [0.956, 0.956, 0.959, 0.958, 0.909]
a_time = [34.08, 36.32, 24.44, 24.91, 37.49]
w_time = [45.88, 18.09, 43.13, 10.31, 25.9]
g_time = [5.92, 3.9, 5.5, 4.44, 66.46]

# plt.plot(a_size, svm_rbf, label='d=123, SVM RBF')
# plt.plot(a_size, svm_linear, label='d=123, SVM Linear')
# plt.plot(a_size, svm_poly, label='d=123, SVM Poly')
# plt.plot(a_size, mlp_lbfgs, label='d=123, MLP lbfgs')
# plt.plot(a_size, mlp_adam, label='d=123, MLP adam')
# plt.plot(w_size, svm_rbf2, label='d=300, SVM RBF')
# plt.plot(w_size, svm_linear2, label='d=300, SVM Linear')
# plt.plot(w_size, svm_poly2, label='d=300, SVM Poly')
# plt.plot(w_size, mlp_lbfgs2, label='d=300, MLP lbfgs')
# plt.plot(w_size, mlp_adam2, label='d=300, MLP adam')
# plt.plot(method, a_accuracy, label='d=123')
# plt.plot(method, w_accuracy, label='d=300')
# plt.plot(method, g_accuracy, label='d=5000')
# plt.plot(method, a_time, label='d=123')
# plt.plot(method, w_time, label='d=300')
# plt.plot(method, g_time, label='d=5000')

train_size = [10000, 20000, 30000, 40000, 50000]
cifar_accuracy = [0.4716, 0.5049, 0.5199, 0.5342, 0.5437]
cifar_time = [406.61, 1501.45, 3132.82, 5406.17, 8369.56]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(train_size, cifar_accuracy, 'g-', label='accuracy', marker='o')
ax2.plot(train_size, cifar_time, 'b-', label='time', marker='o')
# plt.legend(loc='best')
# plt.xlabel('Sample Size of Training Set (%)')
ax1.set_xlabel("Training Size")
# plt.ylabel('Accuracy')
# plt.ylabel('Time (s)')
ax1.set_ylabel("Accuracy", color='g')
ax2.set_ylabel("Time (s)", color='b')
ax1.tick_params(axis='y', colors='g')
ax2.tick_params(axis='y', colors='b')
ax1.spines['left'].set_color('g')
ax2.spines['right'].set_color('b')
# plt.xticks(method)
# plt.xlim(0.5, 2.0)
# plt.ylim(0.92, 1.01)
plt.show()
