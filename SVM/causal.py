import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot

X = np.loadtxt('pair0012.txt')
X = pd.DataFrame(X)

model = lingam.DirectLiNGAM()
model.fit(X)
print(model.causal_order_)
print(model.adjacency_matrix_)

from graphviz import Digraph

dot = Digraph(comment='Causal Discovery')
dot.node('A', 'Age')
dot.node('B', 'Wage per Hour')
dot.edge('A', 'B', '11.06')
dot.view()
