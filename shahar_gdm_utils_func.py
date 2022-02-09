import math
import os.path
import argparse
import numpy as np
import pandas as pd
import networkx as nx


# parser = argparse.ArgumentParser()
# parser.add_argument('--corr_threshold', type=float, default=-1)
# parser.add_argument('--alpha', type=float, default=1)
# args = parser.parse_args()


# load the dataframe (rename some columns)
def load_dataset(ds_path):
    df = pd.read_csv(ds_path, index_col=0)

    translations = {
        'מוצא אם': 'Origin Mother',
        'מוצא אב': 'Origin Father',
        'ארץ לידה': 'Country of Birth',
        'עישון': 'Smoking',
        'גיל': 'Age',
        'גובה': 'Height',
        'לפני הריון': 'before Pregnancy',
        'לפני היריון': 'before Pregnancy',
        'משקל': 'Weight',
        'לחץ דם': 'Blood Pressure',
        'הריון בר סיכון': 'High Risk Pregnancy',
    }

    new_cols = {}

    for i, c in enumerate(df.columns):
        c_new = c
        for t_heb, t_eng in translations.items():
            if t_heb in c:
                c_new = c_new.replace(t_heb, t_eng)

        new_cols[c] = c_new

    df.rename(columns=new_cols, inplace=True)
    return df


# vals_df represents the dataframe with missing values were converted to 0 (the mean)
# existence_df represents the existence of each element in the original dataset (0 or 1)
def find_corr(vals_df, existence_df, alpha=1, corr_threshold=-1):
    # find correlations between the columns of each dataframe
    # nan is got when two columns are the same - replace by 1
    corr_vals_df = vals_df.corr().fillna(1.0)
    corr_existence_df = existence_df.corr().fillna(1.0)

    corr_combined_df = alpha * corr_vals_df + (1 - alpha) * corr_existence_df  # a linear combination of the corralation dataframes

    corr_cols = {}

    # if two columns are correlated (more than corr_threshold) - add them to the dictionary where the key is their correletion
    for i, c in enumerate(corr_combined_df.columns):
        for j in range(i):
            if abs(corr_combined_df.iat[i, j]) > corr_threshold:
                corr_cols[i, j] = corr_combined_df.iat[i, j]

    return corr_cols, corr_combined_df


# convert a sample to a graph
# a nan in the sample is converted to an isolated node, and the others generates a weighted clique where the weights are
# the correlations
# the nodes attributes are their value in the sample (0 in the case of nan)
def graph_from_sample(sample, edge_weights):
    feats = np.nan_to_num(sample)  # convert the nan to 0.0

    G = nx.Graph()

    G.add_nodes_from(range(len(feats)))
    nx.set_node_attributes(G, {i: {"value": v} for i, v in enumerate(feats)})
    G.add_weighted_edges_from([i, j, abs(w)] for (i, j), w in edge_weights.items() if not math.isnan(sample[i]) and
                              not math.isnan(sample[j]))

    return G


# convert each sample in the dataset to a graph
def df_to_graphs(samples_df, edge_weights):
    graphs = {}

    for idx, sample in samples_df.iterrows():
        graphs[idx] = graph_from_sample(sample.values, edge_weights)

    return graphs


# def print_basic_graph_statistics(graphs):
#     nodes_number_dict = {}
#     edges_number_dict = {}
#
#     for graph in graphs.values():
#         node_number = graph.nodes_number_dict()
#         edges_number = graph.number_of_edges()
#         if node_number not in nodes_number_dict:
#             nodes_number_dict[node_number] = 0
#         if edges_number not in edges_number_dict:
#             edges_number_dict[edges_number] = 0
#         nodes_number_dict[node_number] = nodes_number_dict[node_number] + 1
#         edges_number_dict[edges_number] = edges_number_dict[edges_number] + 1
#     print("nodes_number_dict", nodes_number_dict)
#     print("edges_number_dict", edges_number_dict)


if __name__ == '__main__':
    samples_path = os.path.join('week_14_new.csv')
    labels_path = os.path.join('gdm.csv')

    samples_df = load_dataset(samples_path)
    labels = load_dataset(labels_path).values.astype(int)

    vals_df = samples_df.copy(deep=True).fillna(0)
    existence_df = samples_df.copy(deep=True).isnull().astype(int).apply(lambda x: 1 - x)

    corr, corr_combined_df = find_corr(vals_df, existence_df, alpha=args.alpha, corr_threshold=args.corr_threshold)

    graphs = df_to_graphs(samples_df, corr)
    x=1