import os
import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from tqdm import tqdm
from taxonomy_tree_average_sons import create_tax_tree


def create_multi_graph(graphs_list):
    multi_graph = nx.MultiGraph()
    # Assumption: all graphs have the same nodes
    multi_graph.add_nodes_from(graphs_list[0].nodes(data=False))
    for graph in graphs_list:
        multi_graph.add_edges_from(graph.edges())

    multi_graph_adj_mat = nx.adjacency_matrix(multi_graph).todense()
    G = nx.from_numpy_matrix(np.matrix(multi_graph_adj_mat))
    # weighted_graph_adj_mat = nx.adjacency_matrix(G).todense()
    # multi_graph_adj_mat_df = pd.DataFrame(multi_graph_adj_mat)
    # weighted_graph_adj_mat_df = pd.DataFrame(weighted_graph_adj_mat)
    # adj_mat_df.to_csv("multi_graph_adj_mat.csv")
    # print(multi_graph_adj_mat_df.equals(weighted_graph_adj_mat_df))
    return G

def check_multi_graph(graphs_list):
    edge_dict = {}
    for graph in graphs_list:
        cur_edges = graph.edges()
        for source, dest in cur_edges:
            if source not in edge_dict:
                edge_dict[source] = {dest: 0}
            if dest not in edge_dict[source]:
                edge_dict[source][dest] = 0
            edge_dict[source][dest] += 1
    return edge_dict


def node2vec_embed(graph):
    node2vec = Node2Vec(graph, dimensions=128, walk_length=80, num_walks=16, weight_key='weight')
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
    return my_dict, X, graph

def find_embed(graphs_list):
    G = create_multi_graph(graphs_list)
    my_dict, X, graph = node2vec_embed(G)
    return my_dict, X, graph


if __name__ == '__main__':
    data_file_path = os.path.join("split_datasets", 'IBD_split_dataset', 'OTU_IBD_after_mipmlp_Genus.csv')
    microbiome_df = pd.read_csv(data_file_path, index_col='ID')
    graphs = []
    for i, mom in tqdm(enumerate(microbiome_df.iterrows()), desc='Create graphs', total=len(microbiome_df)):
        cur_graph = create_tax_tree(microbiome_df.iloc[i])
        graphs.append(cur_graph)

    edge_dict = check_multi_graph(graphs)
    G = create_multi_graph(graphs)
    my_dict, X, graph = node2vec_embed(G)
    print()
    # count_edges_from_all_graphs(graphs)
