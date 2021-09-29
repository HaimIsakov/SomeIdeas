import os
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from taxonomy_tree_for_pytorch_geometric import create_tax_tree

def create_multi_graph(graphs_list):
    multi_graph = nx.MultiGraph()
    # Assumption: all graphs have the same nodes
    multi_graph.add_nodes_from(graphs_list[0].nodes(data=False))
    for graph in graphs_list:
        multi_graph.add_edges_from(graph.edges())

    multi_graph_adj_mat = nx.adjacency_matrix(multi_graph).todense()
    G = nx.from_numpy_matrix(np.matrix(multi_graph_adj_mat))
    weighted_graph_adj_mat = nx.adjacency_matrix(G).todense()
    multi_graph_adj_mat_df = pd.DataFrame(multi_graph_adj_mat)
    weighted_graph_adj_mat_df = pd.DataFrame(weighted_graph_adj_mat)
    # adj_mat_df.to_csv("multi_graph_adj_mat.csv")
    print(multi_graph_adj_mat_df.equals(weighted_graph_adj_mat_df))

def count_edges_from_all_graphs(graphs_list):
    edges_dict = {}
    for graph in graphs_list:
        cur_edges = dict(graph.edges())
        for k, v in cur_edges.items():
            if k not in edges_dict:
                edges_dict[k] = 0
            edges_dict[k] += 1
    print()


if __name__ == "__main__":
    data_file_path = os.path.join("..", "split_datasets", 'IBD_split_dataset', 'OTU_IBD_after_mipmlp_Genus.csv')
    microbiome_df = pd.read_csv(data_file_path, index_col='ID')
    graphs = []
    for i, mom in tqdm(enumerate(microbiome_df.iterrows()), desc='Create graphs', total=len(microbiome_df)):
        # cur_graph = create_tax_tree(microbiome_df.iloc[i], ignore_values=0, ignore_flag=True)
        cur_graph = create_tax_tree(microbiome_df.iloc[i])
        graphs.append(cur_graph)

    create_multi_graph(graphs)
    # count_edges_from_all_graphs(graphs)
