from __init__ import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from loggers import PrintLogger
from features_algorithms.vertices.motifs import MotifsNodeCalculator

logger = PrintLogger("MyLogger")


def load_graph(path):
    if path.endswith(".pkl"):
        g: nx.Graph = nx.read_gpickle(open(path, 'rb'))
    else:
        df = pd.read_csv(path)
        g: nx.Graph = nx.from_pandas_edgelist(df, source='n1', target='n2')
    return g


def draw_graph(gnx: nx.Graph):
    pos = nx.layout.spring_layout(gnx)
    nx.draw_networkx_nodes(gnx, pos)
    if gnx.is_directed():
        nx.draw_networkx_edges(gnx, pos, arrowstyle='->', arrowsize=30)
    else:
        nx.draw_networkx_edges(gnx, pos)

    nx.draw_networkx_labels(gnx, pos, font_size=10, font_family='sans-serif')

    plt.axis('off')
    plt.show()


def main():
    path = 'test_undirected'
    g = load_graph(path)
    feature = MotifsNodeCalculator(g, level=4, logger=logger)
    feature.build()

    mx = feature.to_matrix(mtype=np.matrix, should_zscore=False)
    print(mx)


if __name__ == '__main__':
    main()
