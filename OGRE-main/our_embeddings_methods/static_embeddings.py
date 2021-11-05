"""
Our suggested static embedding methods full calculation.
"""

import math
from OGRE import *
from D_W_OGRE import main_D_W_OGRE
import time


class StaticEmbeddings:
    """
    Class to run one of our suggested static embedding methods.
    """
    def __init__(self, name, G, initial_size=100, initial_method="node2vec", method="OGRE", H=None,
                 dim=128, choose="degrees", regu_val=0, weighted_reg=False, epsilon=0.1, file_tags=None):
        """
        Init function to initialize the class
        :param name: Name of the graph/dataset
        :param G: Our graph
        :param initial_method: Initial state-of-the-art embedding algorithm for the initial embedding. Options are
                "node2vec" , "gf", "HOPE" or "GCN". Default is "node2vec".
        :param method: One of our suggested static embedding methods. Options are "OGRE", "DOGRE" or "WOGRE". Default
                is "OGRE".
        :param initial_size: Size of initial embedding (integer that is less or equal to the number of nodes in the
                graph). Default value is 100.
        :param H: a networkx graph - If you already have an existing sub graph for the initial embedding, insert it as input as networkx
                  graph (initial_size is not needed in this case), else None.
        :param dim: Embedding dimension. Default is 128.
        :param choose: Weather to choose the nodes of the initial embedding by highest degree or highest k-core score.
                Options are "degrees" for the first and "k-core" for the second. (In our experiments we use degrees).
        :param regu_val: If DOGRE/WOGRE method is applied, one can have a regression with regularization, this is the value
                of the regularization coefficient. Default is 0 (no regularization).
        :param weighted_reg: If DOGRE/WOGRE method is applied, one can have a weighted regression. True for weighted
                regression, else False. Default is False.
        :param epsilon: The weight that is given to the embeddings of the second order neighbours in OGRE (no need in DOGRE/WOGRE).
        :param file tags: If the initial embedding is GCN, file tags is needed. You can see an example format in "labels" directory.
        """
        self.name = name
        # The graph which needs to be embed. If you have a different format change to your own loader.
        self.graph = G
        self.epsilon = epsilon
        self.initial_method = initial_method
        self.embedding_method = method
        if H is None:
            if type(initial_size) == int:
                self.initial_size = [initial_size]
            elif len(initial_size) >= 1:
                self.initial_size = initial_size
            else:
                self.initial_size = calculate_factor_initial_size(self.graph.number_of_nodes(), math.sqrt(10))
        else:
            self.initial_size = [H.number_of_nodes()]
        print("initial size: ", self.initial_size)
        self.dim = dim
        # dictionary of parameters for state-of-the-art method
        self.params_dict = self.define_params_for_initial_method()
        self.choose = choose
        # calculate the given graph embedding and return a dictionary of nodes as keys and embedding vectors as values,
        self.list_dicts_embedding, self.times, self.list_initial_proj_nodes = self.calculate_embedding(regu_val, weighted_reg, epsilon, file_tags, H)
        #self.list_dicts_embedding = []
        #names = []
        #for i in self.initial_size:
        #    names.append("{} + {} + {} + {} + {}".format(name, initial_method, method, str(i), self.epsilon))
            #names.append("{} + {} + {} + {}".format(name, initial_method, method, str(i)))
        #for j in names:
         #   d = load_embedding(os.path.join("..", "embeddings_degrees"), j)
          #  self.list_dicts_embedding.append(d)
        
    def define_params_for_initial_method(self):
        """
        According to the initial state-of-the-art embedding method, create the dictionary of parameters.
        :return: Parameters dictionary
        """
        if self.initial_method == "node2vec":
            params_dict = {"dimension": self.dim, "walk_length": 80, "num_walks": 16, "workers": 2}
        elif self.initial_method == "GF":
            params_dict ={"dimension": self.dim, "eta": 0.1, "regularization": 0.1, "max_iter": 3000, "print_step": 100}
        elif self.initial_method == "HOPE":
            params_dict = {"dimension": self.dim, "beta": 0.1}
        elif self.initial_method == "GCN":
            params_dict = {"dimension": self.dim, "epochs": 100, "lr": 0.1, "weight_decay": 0, "hidden": 2000,
                           "dropout": 0.2}
        else:
            params_dict = None
        return params_dict

    def calculate_embedding(self, regu_val, weighted_reg, epsilon, file_tags=None, H=None):
        """
        Calculate the graph embedding.
        :return: An embedding dictionary where keys are the nodes that are in the final embedding and values are
                their embedding vectors.
        """
        list_dicts, times, list_initial_proj_nodes = main_static(self.name, self.embedding_method, self.initial_method,
                                                                 self.graph, self.initial_size, self.dim, 
                                                                 self.params_dict, self.choose, regu_val=regu_val, 
                                                                 weighted_reg=weighted_reg, epsilon=epsilon, 
                                                                 file_tags=file_tags, F=H)
        return list_dicts, times, list_initial_proj_nodes

    def save_embedding(self, path):
        """
        Save the calculated embedding in a .npy file.
        :param path: Path to where to save the embedding
        :return: The file name
        """
        for j in range(len(self.list_dicts_embedding)):
            dict_embedding = self.list_dicts_embedding[j]
            file_name = self.name + " + " + self.initial_method + " + " + self.embedding_method + " + " \
                        + str(self.initial_size[j]) + " + " + str(self.epsilon)
            np.save(os.path.join(path, '{}.npy'.format(file_name)), dict_embedding)


def add_weights(G):
    """
    If the graph is not weighted, add weights equal to 1.
    :param G: The graph
    :return: The weighted version of the graph
    """
    edges = list(G.edges())
    for e in edges:
        G[e[0]][e[1]] = {"weight": 1}
    return G


def calculate_factor_initial_size(n, key):
    """
    Calculate different initial embedding sizes by a chosen factor.
    :param n: Number of nodes in the graph
    :param key: The factor- for example if key==10, the sizes will be n/10, n/100, n/100, .... the minimum is 100 nodes
    in the initial embedding
    :return: List of initial embedding sizes
    """
    initial = []
    i = n
    while i > 100:
        i = int(i/key)
        i_2 = int(i/2)
        key = pow(key, 2)
        if i > 100:
            initial.append(i)
        if i_2 > 100:
            initial.append(i_2)
    initial.append(100)
    ten_per = int(n/10)
    initial.append(ten_per)
    initial.sort()
    return initial


def load_embedding(path, file_name):
    """
    Given a .npy file - embedding of a given graph. return the embedding dictionary
    :param path: Where this file is saved.
    :param file_name: The name of the file
    :return: Embedding dictionary
    """
    data = np.load(os.path.join(path, '{}.npy'.format(file_name)), allow_pickle=True)
    dict_embedding = data.item()
    return dict_embedding


def main_static(name, method, initial_method, G, initial, dim, params, choose="degrees", regu_val=0., weighted_reg=False,
                epsilon=0.1, file_tags=None, F=None):
    """
    Main function to run our different static embedding methods- OGRE, DOGRE, WOGRE, LGF.
    :param method: One of our methods - OGRE, DOGRE, WOGRE, LGF (string)
    :param initial_method: state-of-the-art algorithm for initial embedding - node2vec, HOPE, GF or GCN (string)
    :param G: The graph to embed (networkx graph)
    :param initial: A list of different sizes of initial embedding (in any length, sizes must be integers).
    :param dim: Embedding dimension (int)
    :param params: Dictionary of parameters for the initial algorithm
    :param choose: How to choose the nodes in the initial embedding - if == "degrees" nodes with highest degrees are
    chosen, if == "k-core" nodes with highest k-core score are chosen.
    :param regu_val: If DOGRE/WOGRE method is applied, one can have a regression with regularization, this is the value
    of the regularization coefficient. Default is 0 (no regularization).
    :param weighted_reg: If DOGRE/WOGRE method is applied, one can have a weighted regression. True for weighted regression,
    else False. Default is False.
    :param epsilon: Determine the weight given to the second order neighbours (only in OGRE method).
    :param file_tags: If the initial embedding is GCN, file tags is needed. You can see an example format in "labels" directory.
    :param F: a networkx graph - If you already have an existing sub graph for the initial embedding, insert it as input as networkx
                  graph (initial_size is not needed in this case), else None.
    :return: - list of embedding dictionaries, each connected to a different initial embedding size. The keys of the
             dictionary are the nodes that in the final embedding, values are the embedding vectors.
             - list of times - each member is the running time of the embedding method, corresponding to the matching
             size of initial embedding.
             - list of nodes that are in the initial embedding.

    """
    if method == "DOGRE":
        list_dicts, times, list_initial_proj_nodes = main_D_W_OGRE(name, G, initial_method, method, initial, dim, 
                                                                   params, choose, regu_val, weighted_reg, 
                                                                   file_tags=file_tags, F=F)
    elif method == "WOGRE":
        list_dicts, times, list_initial_proj_nodes = main_D_W_OGRE(name, G, initial_method, method, initial, dim, 
                                                                   params, choose, regu_val, weighted_reg, 
                                                                   file_tags=file_tags, F=F)

    elif method == "OGRE":
        list_dicts, times, list_initial_proj_nodes = main_OGRE(name, initial_method, G, initial, dim, params,
                                                               choose="degrees", epsilon=epsilon, file_tags=file_tags, 
                                                               F=F)
    else:
        print("Invalid embedding method, calculate with OGRE instead")
        list_dicts, times, list_initial_proj_nodes = main_OGRE(name, initial_method, G, initial, dim, params,
                                                               choose="degrees", epsilon=epsilon, file_tags=file_tags, 
                                                               F=F)
            
    return list_dicts, times, list_initial_proj_nodes


"""
Example code to calculate embedding can be seen in the file- evaluation_tasks/calculate_static_embeddings.py.
"""

# name = "DBLP"  # name
# file_tags = "../labels/dblp_tags.txt"     # labels file
# dataset_path = os.path.join("..", "datasets")     # path to where datasets are saved
# embeddings_path = os.path.join("..", "embeddings_degrees")     # path to where embeddings are saved
# G = nx.read_edgelist(os.path.join(dataset_path, name + ".txt"), create_using=nx.DiGraph(), delimiter=",")    # read the graph
# G = add_weights(G)   # add weights 1 to the graph if it is unweighted
# initial_method = "node2vec"     # known embedding algorithm to embed initial vertices with
# method = "OGRE"        # our method
# initial_size = [100]    # size of initial embedding
# choose = "degrees"     # choose initial nodes by highest degree
# dim = 128     # embedding dimension
# H = None    # put a sub graph if you have an initial sub graph to embed, else None
# SE = StaticEmbeddings(name, G, initial_size, initial_method=initial_method, method=method, H=H, dim=dim,
#                                                           choose=choose, file_tags=file_tags)
# SE.save_embedding(embeddings_path)    # save the embedding
# list_dict_embedding = SE.list_dicts_embedding    # the embedding saved as a dict, this is list of dicts, each dict for different initial size
