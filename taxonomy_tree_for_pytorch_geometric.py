import pickle
# from openpyxl import Workbook, load_workbook
import os
import re
import math
import pandas
import networkx as nx
import pandas as pd
# from torch_geometric.data import DataLoader
# from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

"""
every bacteria is an object to easily store it's information
"""
class Bacteria:
    def __init__(self, string, val):
        string = string.replace(" ", "")
        lst = re.split("; |__|;", string)
        self.val = val
        # removing letters and blank spaces
        for i in range(0, len(lst)):
            if len(lst[i]) < 2:
                lst[i] = 0
        lst = [value for value in lst if value != 0]
        # Default fall value
        if len(lst) == 0:
            lst = ["Bacteria"]
        self.lst = lst


def create_tax_tree(series, zeroflag=True):
    tempGraph = nx.Graph()
    """workbook = load_workbook(filename="random_Otus.xlsx")
    sheet = workbook.active"""
    valdict = {("Bacteria",): 0, ("Archaea",): 0}
    bac = []
    for i, (tax, val) in enumerate(series.items()):
        # adding the bacteria in every column
        bac.append(Bacteria(tax, val))
        if len(bac[i].lst) == 1 and bac[i].lst[0] == "Bacteria":
            valdict[("Bacteria",)] += bac[i].val
        if len(bac[i].lst) == 1 and bac[i].lst[0] == "Archaea":
            valdict[("Archaea",)] += bac[i].val
        # connecting to the root of the tempGraph
        tempGraph.add_edge(("anaerobe",), (bac[i].lst[0],))
        # connecting all levels of the taxonomy
        for j in range(0, len(bac[i].lst) - 1):
            updateval(tempGraph, bac[i], valdict, j, True)
        # adding the value of the last node in the chain
        updateval(tempGraph, bac[i], valdict, len(bac[i].lst) - 1, False)
    valdict[("anaerobe",)] = valdict[("Bacteria",)] + valdict[("Archaea",)]
    return create_final_graph(tempGraph, valdict, zeroflag)


def updateval(graph, bac, vald, num, adde):
    if adde:
        graph.add_edge(tuple(bac.lst[:num+1]), tuple(bac.lst[:num+2]))
    # adding the value of the nodes
    if tuple(bac.lst[:num+1]) in vald:
        vald[tuple(bac.lst[:num+1])] += bac.val
    else:
        vald[tuple(bac.lst[:num+1])] = bac.val


def create_final_graph(tempGraph, valdict, zeroflag):
    graph = nx.Graph()
    for e in tempGraph.edges():
        node1_name = e[0]
        node1_val = valdict[e[0]]
        node2_name = e[1]
        node2_val = valdict[e[1]]
        graph.add_node(node1_name, val=node1_val)
        graph.add_node(node2_name, val=node2_val)
        if not zeroflag or node1_val * node2_val != 0:
            graph.add_edge(node1_name, node2_name)
    return graph


if __name__ == '__main__':
    # data_file_path = os.path.join('Cirrhosis_split_dataset', 'train_val_set_Cirrhosis_microbiome.csv')
    # data_file_path = os.path.join('GDM_split_dataset', 'train_val_set_gdm_microbiome.csv')
    # data_file_path = os.path.join('Allergy', 'OTU_Allergy_after_mipmlp_Genus_same_ids.csv')
    # data_file_path = os.path.join('bw_split_dataset', 'OTU_Black_vs_White_after_mipmlp_Genus_same_ids.csv')
    data_file_path = os.path.join('IBD_split_dataset', 'OTU_IBD_after_mipmlp_Genus.csv')
    microbiome_df = pd.read_csv(data_file_path, index_col='ID')
    nodes_number = []
    graphs = []
    for i, mom in tqdm(enumerate(microbiome_df.iterrows()), desc='Create graphs', total=len(microbiome_df)):
        # cur_graph = create_tax_tree(microbiome_df.iloc[i], ignore_values=0, ignore_flag=True)
        cur_graph = create_tax_tree(microbiome_df.iloc[i])
        graphs.append(cur_graph)
        nodes_number.append(cur_graph.number_of_nodes())
        # print("Nodes Number", cur_graph.number_of_nodes())

    data_list = []
    for g in graphs:
        data = from_networkx(g, group_node_attrs=['val'])  # Notice: convert file was changed explicitly
        data_list.append(data)
    loader = DataLoader(data_list, batch_size=32, exclude_keys=['val'], shuffle=True)

    for step, data in enumerate(loader):
        print(f'Step {step + 1}:')
    print()
