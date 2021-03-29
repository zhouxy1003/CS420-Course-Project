from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame

# generate 2d classification dataset
data, tag = make_blobs(n_samples=100, centers=3, n_features=2, cluster_std=[1.5, 1.5, 1.5])

# scatter plot, dots colored by class value
df = DataFrame(dict(x=data[:, 0], y=data[:, 1], label=tag))

print(df)

f = open("/data2.txt", "w")
for x, y in zip(data[:, 0], data[:, 1]):
    f.write(str(x))
    f.write("\t")
    f.write(str(y))
    f.write('\n')
f.close()
f = open("/tag2.txt", "w")
for z in tag:
    f.write(str(z))
    f.write('\n')
f.close()

colors = {0: 'red', 1: 'blue', 2: 'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')

for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key,
               color=colors[key])

pyplot.show()
