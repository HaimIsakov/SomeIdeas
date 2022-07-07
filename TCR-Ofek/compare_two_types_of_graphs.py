import os
import sys
import numpy as np
import pandas as pd
from colorama import Fore
from matplotlib import pyplot as plt
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import jaccard_score

def create_binary_network():
    # for i, t in enumerate(np.arange(0.5, 1, 0.1)):
    #     print(t)
    t=0.7
    file_corr = os.path.join("networks_to_compare", "corr",
                             f"new_graph_type_corr_tcr_corr_mat_125_with_sample_size_547_run_number_1_mission_concat_graph_and_values.csv")
    file_proj = os.path.join("networks_to_compare", "proj",
                             f"new_graph_type_proj_tcr_mat_125_with_sample_size_547_run_number_1_mission_concat_graph_and_values.csv")

    df_corr = pd.read_csv(file_corr, index_col=0)
    df_proj = pd.read_csv(file_proj, index_col=0)
    df_corr_binary = (np.abs(df_corr) >= t).astype(int)
    df_proj_binary = (np.abs(df_proj) >= t).astype(int)
    print(df_corr_binary.sum().sum())
    print(df_proj_binary.sum().sum())
    print(jaccard_score(df_corr_binary, df_proj_binary, average="micro"))

create_binary_network()

# corr_n_conn, proj_n_conn, jaccard_index = [], [], []
# for i in range(2):
#     file_corr = os.path.join("networks_to_compare", "corr",
#                              f"new_graph_type_corr_tcr_corr_mat_125_with_sample_size_547_run_number_{i}_mission_concat_graph_and_values.csv")
#     file_proj = os.path.join("networks_to_compare", "proj",
#                              f"new_graph_type_proj_tcr_mat_125_with_sample_size_547_run_number_{i}_mission_concat_graph_and_values.csv")
#
#     df_corr = pd.read_csv(file_corr, index_col=0)
#     df_proj = pd.read_csv(file_proj, index_col=0)
#
#     df_corr_values = np.array(df_corr.values).astype(int)
#     df_proj_values = np.array(df_proj.values).astype(int)
#
#     G_corr = nx.from_numpy_matrix(df_corr_values)
#     G_proj = nx.from_numpy_matrix(df_proj_values)
#     corr_n_conn.append(df_corr_values.sum().sum())
#     proj_n_conn.append(df_proj_values.sum().sum())
#
#     print("Number of connections - corr", corr_n_conn[-1])
#     print("Number of connections - proj", proj_n_conn[-1])
#     jaccard_index.append(jaccard_score(df_corr_values, df_proj_values, average="micro"))
#     print(jaccard_index[-1])
# print(np.mean(corr_n_conn))
# print(np.mean(proj_n_conn))
# print(np.mean(jaccard_index))
#
# x=2
