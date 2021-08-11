import os

import networkx as nx
import matplotlib.pyplot as plt
import re

import pandas as pd
from tqdm import tqdm


class Bacteria:
    def __init__(self, string, val):
        lst = re.split("; |__| ", string)
        self.val = val
        # removing letters and blank spaces
        for i in range(0, len(lst)):
            if len(lst[i]) < 2:
                lst[i] = 0
        lst = [value for value in lst if value != 0]
        self.lst = lst


# function from Ariel Rozen
def create_tax_tree(series, ignore_values=-1, ignore_flag=False):
    tempGraph = nx.Graph()
    """workbook = load_workbook(filename="random_Otus.xlsx")
    sheet = workbook.active"""
    valdict = {}
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
    return create_final_graph(tempGraph, valdict, ignore_values, ignore_flag)


def updateval(graph, bac, vald, num, adde):
    if adde:
        graph.add_edge(tuple(bac.lst[:num + 1]), tuple(bac.lst[:num + 2]))
    # adding the value of the nodes
    if tuple(bac.lst[:num + 1]) in vald:
        vald[tuple(bac.lst[:num + 1])] += bac.val
    else:
        vald[tuple(bac.lst[:num + 1])] = bac.val


def create_final_graph(tempGraph, valdict, ignore_values, ignore_flag):
    graph = nx.Graph()
    for e in tempGraph.edges():
        # אם לא צריך להתעלם מאף אחד - תוסיף את כולם - אין בעיה, או אם צריך להתעלם אז תוסיף רק את אלה שהם לא מהערכים שצריך להתעלם מהם
        if not ignore_flag or (ignore_flag and (valdict[e[0]] != ignore_values and valdict[e[1]] != ignore_values)):
            graph.add_edge((e[0], valdict[e[0]]), (e[1], valdict[e[1]]))
    return graph


def draw_tree(graph, threshold=0.0):
    if type(threshold) == tuple:
        lower_threshold, higher_threshold = threshold
    else:
        lower_threshold, higher_threshold = -threshold, threshold
    labelg = {}
    labelr = {}
    colormap = []
    sizemap = []
    for node in graph:
        if node[0] == "base":
            colormap.append("white")
            sizemap.append(0)
        else:
            if node[1] < lower_threshold:
                colormap.append("red")
                labelr[node] = node[0][-1]
            elif node[1] > higher_threshold:
                colormap.append("green")
                labelg[node] = node[0][-1]
            else:
                colormap.append("yellow")
            sizemap.append(node[1] / 1000 + 5)
    # drawing the graph
    # pos = graphviz_layout(graph, prog="twopi", root="base")
    pos = nx.spring_layout(graph)
    lpos = {}
    for key, loc in pos.items():
        lpos[key] = (loc[0], loc[1] + 0.02)
    nx.draw(graph, pos, node_size=sizemap, node_color=colormap, width=0.3)
    nx.nx.draw_networkx_labels(graph, lpos, labelr, font_size=7, font_color="red")
    nx.nx.draw_networkx_labels(graph, lpos, labelg, font_size=7, font_color="green")
    plt.draw()
    plt.show()
    plt.savefig("taxtree.png")

if __name__ == '__main__':
    data_file_path = os.path.join('Cirrhosis_split_dataset', 'train_val_set_Cirrhosis_microbiome.csv')
    microbiome_df = pd.read_csv(data_file_path, index_col='ID')
    for i, mom in tqdm(enumerate(microbiome_df.iterrows()), desc='Create graphs'):
        cur_graph = create_tax_tree(microbiome_df.iloc[i], ignore_values=0, ignore_flag=True)
