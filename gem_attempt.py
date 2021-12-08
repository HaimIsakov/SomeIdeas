import os

import numpy as np
import pandas as pd
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
# from time import time
import seaborn as sns
from gem.embedding.gf import GraphFactorization
from gem.embedding.hope import HOPE
from gem.embedding.lap import LaplacianEigenmaps
from gem.embedding.lle import LocallyLinearEmbedding
# from gem.embedding.node2vec import node2vec
# from gem.embedding.sdne import SDNE
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from node2vec_embed import create_multi_graph
from taxonomy_tree_average_sons import create_tax_tree

if __name__ == '__main__':
    data_file_path = os.path.join("split_datasets", 'Cirrhosis_split_dataset', 'train_val_set_Cirrhosis_microbiome.csv')
    microbiome_df = pd.read_csv(data_file_path, index_col='ID')
    graphs = []
    for i, mom in tqdm(enumerate(microbiome_df.iterrows()), desc='Create graphs', total=len(microbiome_df)):
        cur_graph = create_tax_tree(microbiome_df.iloc[i])
        graphs.append(cur_graph)

    G = create_multi_graph(graphs)
    # G = G.to_directed()
    embedding = HOPE(d=128, beta=0.01)
    # embedding = GraphFactorization(d=128, max_iter=1000, eta=1 * 10 ** -4, regu=1.0, data_set=None)
    # embedding = LaplacianEigenmaps(d=128)
    # embedding = LocallyLinearEmbedding(d=128)
    embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
    X = embedding.get_embedding()
    X_embedded = TSNE(n_components=2).fit_transform(np.asarray(X, dtype='float64'))
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], legend='full')
    plt.show()
    print()
