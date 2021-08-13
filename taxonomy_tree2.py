# from openpyxl import Workbook, load_workbook
import os
import re
import math
import pandas
import networkx as nx
import pickle

import pandas as pd
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
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
        if not zeroflag or valdict[e[0]] * valdict[e[1]] != 0:
            graph.add_edge((e[0], valdict[e[0]]),
                           (e[1], valdict[e[1]]))
        else:
            graph.add_node((e[0], valdict[e[0]]))
            graph.add_node((e[1], valdict[e[1]]))
    return graph


def draw_tree(graph, threshold=0.0):
    labelg = {}
    labelr = {}
    colormap = []
    sizemap = []
    for node in graph:
        if node[0] == "base":
            colormap.append("white")
            sizemap.append(0)
        else:
            if node[1] < -threshold:
                colormap.append("red")
                labelr[node] = node[0][-1]
            elif node[1] > threshold:
                colormap.append("green")
                labelg[node] = node[0][-1]
            else:
                colormap.append("yellow")
            sizemap.append(node[1] / 1000 + 5)
    # drawing the graph
    pos = graphviz_layout(graph, prog="twopi", root="base")
    #pos = nx.spring_layout(graph)
    lpos = {}
    for key, loc in pos.items():
        lpos[key] = (loc[0], loc[1] + 0.02)
    nx.draw(graph, pos, node_size=sizemap, node_color=colormap, width=0.3)
    nx.nx.draw_networkx_labels(graph, lpos, labelr, font_size=7, font_color="red")
    nx.nx.draw_networkx_labels(graph, lpos, labelg, font_size=7, font_color="green")
    plt.draw()
    plt.savefig("taxtree.png")


if __name__ == '__main__':

    # data_file_path = os.path.join('Cirrhosis_split_dataset', 'train_val_set_Cirrhosis_microbiome.csv')
    # data_file_path = os.path.join('GDM_split_dataset', 'train_val_set_gdm_microbiome.csv')
    # data_file_path = os.path.join('Allergy', 'OTU_Allergy_after_mipmlp_Genus_same_ids.csv')
    # data_file_path = os.path.join('Black_vs_White_split_dataset', 'OTU_Black_vs_White_after_mipmlp_Genus_same_ids.csv')
    data_file_path = os.path.join('IBD_split_dataset', 'OTU_IBD_after_mipmlp_Genus.csv')
    microbiome_df = pd.read_csv(data_file_path, index_col='ID')
    nodes_number = []
    graphs = []
    for i, mom in tqdm(enumerate(microbiome_df.iterrows()), desc='Create graphs', total=len(microbiome_df)):
        # cur_graph = create_tax_tree(microbiome_df.iloc[i], ignore_values=0, ignore_flag=True)
        cur_graph = create_tax_tree(microbiome_df.iloc[i])
        graphs.append(cur_graph)
        nodes_number.append(cur_graph.number_of_nodes())
        # print("Number of Nodes", cu)
        # print(cur_graph.nodes())
        # print(cur_graph.number_of_nodes())

    def sort_all_graphs(graphs_list):
        temp_graph_list = []
        for graph in graphs_list:
            temp_graph = nx.Graph()
            temp_graph.add_nodes_from(sorted(graph.nodes(data=True)))
            temp_graph.add_edges_from(graph.edges(data=True))
            # temp_graph.add_edges_from(sorted(graph.edges(data=True)))
            temp_graph_list.append(temp_graph)
        return temp_graph_list
    graphs_list = sort_all_graphs(graphs)
    for i in graphs_list:
        values = [node_name for node_name, value in i.nodes()]
        # print(values)
        # print("----------------------------------------------")
    print(all(x==nodes_number[0] for x in nodes_number))
