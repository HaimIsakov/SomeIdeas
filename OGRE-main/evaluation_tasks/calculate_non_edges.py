"""
For link prediction mission, one needs a csv file of non edges of each graph. This file creates them.
For your own dataset, you need its name and path to where it can be found (see downward). If your dataset needs a
different way of loading, add it, in the end you need a networkx graph.
"""

from eval_utils import load_graph
import os
import itertools as IT
import csv
import random
import __init__

name1 = "DBLP"
name2 = "Cora"
name3 = "Pubmed"
name4 = "Yelp"
name5 = "Reddit"
name6 = "Youtube"
name7 = "Flickr"

names = [name1]

datasets_path = os.path.join("..", "datasets")
labels_path = os.path.join("..", "labels")

# How many vertices to choose randomaly and from them create false edges. For small datasets (DBLP, Pubmed, Cora) it can be equal
# to the number of vertices. In large graphs such as Yelp and Reddit, this file may be too large if all false edges are created,
# therefore one should choose the maximal number of vertices that the csv file with their false edges will be the maximal size as possible.
# depending on your server memory and cpu.
number = 15000

for name in names:
    G = load_graph(datasets_path, name, is_weighted=False)

    list_in_embd = list(G.nodes())

    # choosing randomly indices of the nodes because non-edge is too big for all nodes, choose the biggest size your device
    # can handle with
    indexes = random.sample(range(1, len(list_in_embd)), number)
    new_list = []
    for l in indexes:
        new_list.append(list_in_embd[l])
    sub_G = G.subgraph(new_list)
    print(sub_G.number_of_nodes())
    # create a list of all missing edges of the nodes that were chosen randomly
    missing = [pair for pair in IT.combinations(sub_G.nodes(), 2) if not sub_G.has_edge(*pair)]
    print(len(missing))
    print(1)

    # extract list to a csv file, save it where you want, you will need it for link prediction task
    csvfile = open('non_edges_{}.csv'.format(name), 'w', newline='')
    obj = csv.writer(csvfile)
    obj.writerows(missing)
    csvfile.close()

