"""
Implantation of four state-of-the-art static embedding algorithms: Node2Vec, Graph Factorization and HOPE.
Implementations where taken from GEM package [https://github.com/palash1992/GEM] and GCN pytorch implementation 
[https://github.com/tkipf/pygcn].
"""

import networkx as nx
import numpy as np
from node2vec import Node2Vec
from scipy.sparse import identity
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import svds
import os
import time
from gcn.utils import *
from gcn.models import GCN
import torch.nn.functional as F
import torch.optim as optim


class StaticGraphEmbedding:
    def __init__(self, d, method_name, graph):
        """
        Initialize the Embedding class
        :param d: dimension of embedding
        """
        self._d = d
        self._method_name = method_name
        self._graph = graph
        self._dict_embedding = {}

    @staticmethod
    def get_method_name(self):
        """
        Returns the name for the embedding method
        :param self:
        :return: The name of embedding
        """
        return self._method_name

    def learn_embedding(self):
        """
        Learning the graph embedding from the adjacency matrix.
        :param graph: the graph to embed in networkx DiGraph format
        :return:
        """
        pass

    @staticmethod
    def get_embedding(self):
        """
        Returns the learnt embedding
        :return: A numpy array of size #nodes * d
        """
        pass
    

def train_(epoch, num_of_epochs, model, optimizer, features, labels, adj, idx_train, idx_val):
    """
    Train function for GCN.
    """
    t = time.time()
    model.train()
    optimizer.zero_grad()
    x, output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()), 'time: {:.4f}s'.format(time.time() - t))

    if epoch == num_of_epochs - 1:
        return output  # layer after softmax
        #return x  # layer before softmax
    
class GCNModel(StaticGraphEmbedding):
    def __init__(self, name, params, method_name, graph, file_tags):
        super(GCNModel, self).__init__(params["dimension"], method_name, graph)
        # Training settings
        self._adj, self._features, self._labels, self._idx_train, self._idx_val, self._idx_test = \
            new_load_data(graph, file_tags, len(graph.nodes()), name=name)
        self._seed = 42
        self._epochs = params["epochs"]
        self._lr = params["lr"]
        self._weight_decay = params["weight_decay"]
        self._hidden = params["hidden"]
        self._dropout = params["dropout"]
        self._model = GCN(nfeat=self._features.shape[1], nhid=self._hidden, nclass=self._d, dropout=self._dropout)
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay)

    def learn_embedding(self):
        for epoch in range(self._epochs):
            output1 = train_(epoch, self._epochs, self._model, self._optimizer, self._features, self._labels, self._adj, self._idx_train, self._idx_val)
        y = output1.detach().numpy()
        nodes = list(self._graph.nodes())
        self._dict_embedding = {nodes[i]: y[i, :] for i in range(len(nodes))}
        return self._dict_embedding

class GraphFactorization(StaticGraphEmbedding):
    """
    Graph Factorization factorizes the adjacency matrix with regularization.
    Args: hyper_dict (object): Hyper parameters.
    """

    def __init__(self, params, method_name, graph):
        super(GraphFactorization, self).__init__(params["dimension"], method_name, graph)
        """
        Initialize the GraphFactorization class
        Args: params:
            d: dimension of the embedding
            eta: learning rate of sgd
            regu: regularization coefficient of magnitude of weights
            max_iter: max iterations in sgd
            print_step: #iterations to log the prgoress (step%print_step)
        """
        self._eta = params["eta"]
        self._regu = params["regularization"]
        self._max_iter = params["max_iter"]
        self._print_step = params["print_step"]
        self._X = np.zeros(shape=(len(list(self._graph.nodes())), self._d))

    def get_f_value(self):
        """
        Get the value of f- the optimization function
        """
        nodes = list(self._graph.nodes())
        new_names = list(np.arange(0, len(nodes)))
        mapping = {}
        for i in new_names:
            mapping.update({nodes[i]: str(i)})
        H = nx.relabel.relabel_nodes(self._graph, mapping)
        f1 = 0
        for i, j, w in H.edges(data='weight', default=1):
            f1 += (w - np.dot(self._X[int(i), :], self._X[int(j), :])) ** 2
        f2 = self._regu * (np.linalg.norm(self._X) ** 2)
        return H, [f1, f2, f1 + f2]

    def learn_embedding(self):
        """
        Apply graph factorization embedding
        """
        t1 = time.time()
        node_num = len(list(self._graph.nodes()))
        self._X = 0.01 * np.random.randn(node_num, self._d)
        for iter_id in range(self._max_iter):
            my_f = self.get_f_value()
            count = 0
            if not iter_id % self._print_step:
                H, [f1, f2, f] = self.get_f_value()
                print('\t\tIter id: %d, Objective: %g, f1: %g, f2: %g' % (
                    iter_id,
                    f,
                    f1,
                    f2
                ))
            for i, j, w in H.edges(data='weight', default=1):
                if j <= i:
                    continue
                term1 = -(w - np.dot(self._X[int(i), :], self._X[int(j), :])) * self._X[int(j), :]
                term2 = self._regu * self._X[int(i), :]
                delPhi = term1 + term2
                self._X[int(i), :] -= self._eta * delPhi
            if count > 30:
                break
        t2 = time.time()
        projections = {}
        nodes = list(self._graph.nodes())
        new_nodes = list(H.nodes())
        for j in range(len(nodes)):
            projections.update({nodes[j]: self._X[int(new_nodes[j]), :]})
        # X is the embedding matrix and projections are the embedding dictionary
        return self._X, (t2 - t1), projections

    def get_embedding(self):
        return self._X


class HOPE(StaticGraphEmbedding):
    def __init__(self, params, method_name, graph):
        super(HOPE, self).__init__(params["dimension"], method_name, graph)
        """
        Initialize the HOPE class
        Args:
            d: dimension of the embedding
            beta: higher order coefficient
        """
        self._beta = params["beta"]

    def learn_embedding(self):
        """
        Apply HOPE embedding
        """
        A = nx.to_scipy_sparse_matrix(self._graph, format='csc')
        I = identity(self._graph.number_of_nodes(), format='csc')
        M_g = I - - self._beta * A
        M_l = self._beta * A
        # A = nx.to_numpy_matrix(self._graph)
        # M_g = np.eye(len(self._graph.nodes())) - self._beta * A
        # M_l = self._beta * A
        # S = inv(M_g).dot(M_l)
        S = np.dot(inv(M_g), M_l)

        u, s, vt = svds(S, k=self._d // 2)
        X1 = np.dot(u, np.diag(np.sqrt(s)))
        X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
        self._X = np.concatenate((X1, X2), axis=1)

        p_d_p_t = np.dot(u, np.dot(np.diag(s), vt))
        eig_err = np.linalg.norm(p_d_p_t - S)
        print('SVD error (low rank): %f' % eig_err)

        # create dictionary of nodes
        nodes = list(self._graph.nodes())
        projections = {}
        for i in range(len(nodes)):
            y = self._X[i]
            y = np.reshape(y, newshape=(1, self._d))
            projections.update({nodes[i]: y[0]})
        # X is the embedding matrix, S is the similarity, projections is the embedding dictionary
        return projections, S, self._X, X1, X2

    def get_embedding(self):
        return self._X


class NODE2VEC(StaticGraphEmbedding):
    """
    Nod2Vec Embedding using random walks
    """
    def __init__(self, params, method_name, graph):
        super(NODE2VEC, self).__init__(params["dimension"], method_name, graph)
        """
        parameters:
        "walk_length" - Length of each random walk
        "num_walks" - Number of random walks from each source nodes
        "workers" - How many times repeat this process
        """
        self._walk_length = params["walk_length"]
        self._num_walks = params["num_walks"]
        self._workers = params["workers"]

    def learn_embedding(self):
        """
        Apply Node2Vec embedding
        """
        node2vec = Node2Vec(self._graph, dimensions=self._d, walk_length=self._walk_length,
                            num_walks=self._num_walks, workers=self._workers)
        model = node2vec.fit()
        nodes = list(self._graph.nodes())
        self._my_dict = {}
        for node in nodes:
            try:
                self._my_dict.update({node: np.asarray(model.wv.get_vector(node))})
            except KeyError:
                self._my_dict.update({node: np.asarray(model.wv.get_vector(str(node)))})
        self._X = np.zeros((len(nodes), self._d))
        for i in range(len(nodes)):
            try:
                self._X[i, :] = np.asarray(model.wv.get_vector(nodes[i]))
            except KeyError:
                self._X[i, :] = np.asarray(model.wv.get_vector(str(nodes[i])))
        # X is the embedding matrix and projections are the embedding dictionary
        return self._X, self._my_dict

    def get_embedding(self):
        return self._X, self._my_dict


def save_embedding_state_of_the_art(path, dict_embedding, name, initial_method):
    """
    Save the embedding as .npy file format.
    :param path: Path to where to save the embedding
    :param dict_embedding: Embedding dict
    :param name: Name of the dataset
    :param initial_method: State-of-the-art method- "node2vec", "HOPE" or "GF".
    :return:
    """
    file_name = name + " + " + initial_method
    np.save(os.path.join(path, '{}.npy'.format(file_name)), dict_embedding)


def final(name, G, method_name, params, file_tags=None):
    """
    Final function to apply state-of-the-art embedding methods
    :param G: Graph to embed
    :param method_name: state-of-the-art embedding algorithm
    :param params: Parameters dictionary according to the embedding method
    :return: Embedding matrix, embedding dict and running time
    """
    if method_name == "HOPE":
        t = time.time()
        embedding = HOPE(params, method_name, G)
        projections, S, _, X1, X2 = embedding.learn_embedding()
        X = embedding.get_embedding()
        elapsed_time = time.time() - t
        return X, projections, elapsed_time
    elif method_name == "node2vec":
        t = time.time()
        embedding = NODE2VEC(params, method_name, G)
        embedding.learn_embedding()
        X, projections = embedding.get_embedding()
        elapsed_time = time.time() - t
        return X, projections, elapsed_time
    elif method_name == "GF":
        t = time.time()
        embedding = GraphFactorization(params, method_name, G)
        _, _, projections = embedding.learn_embedding()
        X = embedding.get_embedding()
        elapsed_time = time.time() - t
        return X, projections, elapsed_time
    elif method_name == "GCN":
        t = time.time()
        embedding = GCNModel(name, params, method_name, G, file_tags)
        projections = embedding.learn_embedding()
        elapsed_time = time.time() - t
        return None, projections, elapsed_time
    else:
        print("Method is not valid. Valid methods are: node2vec, hope, graph_factorization")
        return None, None, None
