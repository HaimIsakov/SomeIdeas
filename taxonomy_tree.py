# from openpyxl import Workbook, load_workbook
import re
import math
import pandas
import networkx as nx
import pickle

#
# """
# every bacteria is an object to easily store it's information
# """
# class Bacteria:
#     def __init__(self, string, val):
#         string = string.replace(" ", "")
#         lst = re.split(";|__", string)
#         self.val = val
#         # removing letters and blank spaces
#         for i in range(0, len(lst)):
#             if len(lst[i]) < 2:
#                 lst[i] = 0
#         lst = [value for value in lst if value != 0]
#         self.lst = lst
#
#
# def create_tax_tree(series, flag=None, keepFlagged=False):
#     graph = nx.Graph()
#     """workbook = load_workbook(filename="random_Otus.xlsx")
#     sheet = workbook.active"""
#     graph.add_node(("Bacteria",), val=0)
#     graph.add_node(("Archaea",), val=0)
#     bac = []
#     for i, (tax, val) in enumerate(series.items()):
#         # adding the bacteria in every column
#         bac.append(Bacteria(tax, val))
#         # connecting to the root of the tempGraph
#         graph.add_edge(("Anaerobe",), (bac[i].lst[0],))
#         # connecting all levels of the taxonomy
#         for j in range(0, len(bac[i].lst) - 1):
#             updateval(graph, bac[i], j, True)
#         # adding the value of the last node in the chain
#         updateval(graph, bac[i], len(bac[i].lst) - 1, False)
#     graph.nodes[("Anaerobe",)]["val"] = graph.nodes[("Bacteria",)]['val']+graph.nodes[("Archaea",)]['val']
#     return create_final_graph(graph, flag, keepFlagged)
#
#
# def updateval(graph, bac, num, adde):
#     if adde:
#         if tuple(bac.lst[:num+1]) not in graph:
#             graph.add_node(tuple(bac.lst[:num+1]), val=0)
#         if tuple(bac.lst[:num+2]) not in graph:
#             graph.add_node(tuple(bac.lst[:num+2]), val=0)
#
#         graph.add_edge(tuple(bac.lst[:num+1]), tuple(bac.lst[:num+2]))
#
#     new_val = graph.nodes[tuple(bac.lst[:num+1])]['val'] + bac.val
#     # set values
#     graph.nodes[tuple(bac.lst[:num+1])]['val'] = new_val
#
#
#
# def create_final_graph(graph, flag, keepFlagged):
#     for e in graph.edges():
#         if flag is not None and (graph.nodes[e[0]]["val"] == flag or graph.nodes[e[1]]["val"] == flag):
#             graph.remove_edge(*e)
#     if not keepFlagged:
#         graph.remove_nodes_from(list(nx.isolates(graph)))
#     return graph


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
        tempGraph.add_edge("anaerobe", (bac[i].lst[0],))
        # connecting all levels of the taxonomy
        for j in range(0, len(bac[i].lst) - 1):
            updateval(tempGraph, bac[i], valdict, j, True)
        # adding the value of the last node in the chain
        updateval(tempGraph, bac[i], valdict, len(bac[i].lst) - 1, False)
    valdict["anaerobe"] = valdict[("Bacteria",)] + valdict[("Archaea",)]
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


if __name__ == "__main__":
    graph = create_tax_tree(pickle.load(open("graph152forAriel.p", "rb")), flag=0, keepFlagged=True)
    print()
