import numpy as np
from node2vec import Node2Vec


def node2vec_embed(graph):

    node2vec = Node2Vec(graph, dimensions=128, walk_length=80, num_walks=16, workers=2)
    model = node2vec.fit()
    nodes = list(graph.nodes())
    my_dict = {}
    for node in nodes:
        try:
            my_dict.update({node: np.asarray(model.wv.get_vector(node))})
        except KeyError:
            my_dict.update({node: np.asarray(model.wv.get_vector(str(node)))})
    X = np.zeros((len(nodes), 128))
    for i in range(len(nodes)):
        try:
            X[i, :] = np.asarray(model.wv.get_vector(nodes[i]))
        except KeyError:
            X[i, :] = np.asarray(model.wv.get_vector(str(nodes[i])))
    # X is the embedding matrix and projections are the embedding dictionary
    return my_dict, graph

if __name__ == '__main__':
