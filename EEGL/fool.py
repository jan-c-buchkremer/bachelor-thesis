from Framework import *
import networkx as nx
import matplotlib.pyplot as plt

G180 = loadGraphFromPickle('G180')
pos = nx.kamada_kawai_layout(G180)
nx.draw(G180, pos, with_labels=True, node_color='red', node_size=50, edge_color='gray')
plt.show()

