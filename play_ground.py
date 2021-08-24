import numpy as np

from graph_measures.features_algorithms.vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from graph_measures.features_algorithms.vertices.closeness_centrality import ClosenessCentralityCalculator
from graph_measures.features_algorithms.vertices.communicability_betweenness_centrality import \
    CommunicabilityBetweennessCentralityCalculator
from graph_measures.features_algorithms.vertices.general import GeneralCalculator
from graph_measures.features_algorithms.vertices.load_centrality import LoadCentralityCalculator
from graph_measures.features_infra.feature_calculators import FeatureMeta
from graph_measures.loggers import PrintLogger
from taxonomy_tree_for_pytorch_geometric import create_tax_tree
import os
import pandas as pd
from torch_geometric.data import DataLoader
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm
from graph_measures.features_infra.graph_features import GraphFeatures

from graph_measures.features_algorithms.vertices.louvain import LouvainCalculator
from graph_measures.features_algorithms.vertices.betweenness_centrality import BetweennessCentralityCalculator

if __name__ == '__main__':
    # data_file_path = os.path.join('Cirrhosis_split_dataset', 'train_val_set_Cirrhosis_microbiome.csv')
    # data_file_path = os.path.join('GDM_split_dataset', 'train_val_set_gdm_microbiome.csv')
    # data_file_path = os.path.join('Allergy', 'OTU_Allergy_after_mipmlp_Genus_same_ids.csv')
    data_file_path = os.path.join('Black_vs_White_split_dataset', 'OTU_Black_vs_White_after_mipmlp_Genus_same_ids.csv')
    # data_file_path = os.path.join('IBD_split_dataset', 'OTU_IBD_after_mipmlp_Genus.csv')
    microbiome_df = pd.read_csv(data_file_path, index_col='ID')
    nodes_number = []
    graphs = []
    for i, mom in tqdm(enumerate(microbiome_df.iterrows()), desc='Create graphs', total=len(microbiome_df)):
        # cur_graph = create_tax_tree(microbiome_df.iloc[i], ignore_values=0, ignore_flag=True)
        cur_graph = create_tax_tree(microbiome_df.iloc[i])
        graphs.append(cur_graph)
        nodes_number.append(cur_graph.number_of_nodes())
        # print("Nodes Number", cur_graph.number_of_nodes())
    logger = PrintLogger("MyLogger")

    for ind, graph in enumerate(graphs):
        print(ind)
        features_meta = {
            "general": FeatureMeta(GeneralCalculator, {"general"}),
            "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"nd_avg"}),
            "louvain": FeatureMeta(LouvainCalculator, {"lov"}),
            "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
            "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load"}),
            "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"})
            # "communicability_betweenness_centrality": FeatureMeta(CommunicabilityBetweennessCentralityCalculator, {"comm_betweenness"})
        }
        # Hold the set of features as written here.

        features = GraphFeatures(graph, features_meta, dir_path="stamdir", logger=logger)
        features.build()
        # raise Exception()
        mx = features.to_matrix(mtype=np.matrix)
        mx_dict = features.to_dict()
        for node, feature_matrix in mx_dict.items():
            feature_matrix_0 = feature_matrix.tolist()[0]  # the first row in feature_matrix
            for ind, feature in enumerate(feature_matrix_0):
                cur_feature_name = f"feature{ind}"
                graph.nodes[node][cur_feature_name] = feature

        nodes_and_values = graph.nodes(data=True)
        values_matrix = [[feature_value for feature_name, feature_value in value_dict.items()] for node, value_dict in nodes_and_values]
        print()
    # data_list = []
    # for g in graphs:
    #     data = from_networkx(g, group_node_attrs=['val'])  # Notice: convert file was changed explicitly
    #     data_list.append(data)
    # loader = DataLoader(data_list, batch_size=32, exclude_keys=['val'], shuffle=True)
    #
    # for step, data in enumerate(loader):
    #     print(f'Step {step + 1}:')
    # print()
