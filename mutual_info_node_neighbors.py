import os
import sys
for path_name in [os.path.join(os.path.dirname(__file__)),
                  os.path.join(os.path.dirname(__file__), 'Data'),
                  os.path.join(os.path.dirname(__file__), 'Missions')]:
    sys.path.append(path_name)

import pandas as pd
from tqdm import tqdm
from taxonomy_tree_average_sons import *
from sklearn.feature_selection import mutual_info_classif as MIC
from networkx import all_neighbors

if __name__ == "__main__":
    data_file_path = os.path.join("Data", 'split_datasets_new', "bw_split_dataset",
                                  'train_val_set_bw_microbiome.csv')
    microbiome_df = pd.read_csv(data_file_path, index_col='ID')
    nodes_number = []
    graphs = []
    for i, mom in tqdm(enumerate(microbiome_df.iterrows()), desc='Create graphs', total=len(microbiome_df)):
        cur_graph = create_tax_tree(microbiome_df.iloc[i])
        graphs.append(cur_graph)
        break
    graph = graphs[0]
    all_neighbors(graph, node)
    mutual_information = MIC(X, y)