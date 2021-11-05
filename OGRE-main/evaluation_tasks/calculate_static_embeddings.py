"""
Main file to calculate the embeddings with OGRE/DOGRE/WOGRE, and performing link prediction and node classification task.

In order to calculate the embedding, you first must have an edge list file:
"datasets/name_of_dataset.txt" - An edge list txt file. If the graph is unweighted it consists of 2 columns: source, target (with no title, source and target share an edge).
If the graph is weighted, it consists of 3 columns: source target weight. 
Example for unweighted graph:
1 2
2 3
1 4
1 3
Example for weighted graph:
1 2 3
1 3 0.3
1 4 4.5
2 4 0.98
You can see examples for this format in "datasets" directory.

If you want to peform vertex classification task or GCN is your initial embedding, you must have labels file:
"labels/{name_of_dataset}_tags.txt" - A txt file which consists of 2 columns: node, label (no title). Notice all node must have labels!
Example:
1 0
2 0
3 1
4 2

Another possibilty is having a .mat file as in NRL_Benchmark (https://pages.github.com/). In this link, go to either `node classification`
or `link prediction` directories, where a link to datasets you can use in .mat format is avaliable. Then this .mat file is both the
edges and labels file.

If you want to perform link prediction task, you must have non edges file:
"evaluation_tasks/non_edges_{name_of_dataset}" - A csv file which consists of two columns: node1, node2 ; where there is no edge between them (again no title).
In order to produce such file, you can go to evaluation_tasks -> calculate_non_edges.py , and follow the instructions there.

When you have all the files you need (depending on what you want to perform), you can run this file.
1. First initialize DATASET parameters dict:
- name: Name of dataset (as the name of the edge list txt file) (string)
- initial_size: List of initial core sizes. (list)
- dim: Embedding dimension (int)
- is_weighted: True if the graph is weighted, else False (bool)
- choose: "degrees" if the vertices of the initial core are the ones with highest degree (as done in our experiments), else "k_core" if the vertices of the initial core are
the ones with highest k-core score. (string)
- "s_a": True if you also want to calculate state-of-the-art embeddings (node2vec/GF/HOPE/GCN), else False.
Params for OGRE:
- epsilon: Weight to the second order neighbours embedding. For more details you can go to the implementation- our_embedding_methods -> OGRE.py (float).
Params for DOGRE/WOGRE:
- "regu_val": Regularization value for regression, only for DOGRE/WOGRE. For more details you can go to the implementation- our_embedding_methods -> D_W_OGRE.py (float).
- "weighted_reg": True for weighted regression, else False.
If the initial embedding method is GCN and/or a vertex classification task is applied, a labels file is also necessary:
- "label_file": path and name (together), so it can be read directly.
2. methods_ : List of our suggested embedding methods (OGRE/DOGRE/WOGRE) with whom you want to embed the given graph. 
3. initial_methods_ : List of state-of-the-art embedding methods (node2vec/GF/HOPE/GCN) with whom the initial core will be embed.
4. params_dict_ : Parameters for state-of-the-art embeddings. These are the optimal ones (according to their papers). For more details you can go to- 
state_of_the_art -> state_of_the_art_embedding.py
5. save_: True if you want to save the embedding in a .npy format, else False.

Once you have that, you can run "calculate_static_embeddings" function to get the embeddings as dictionaries. You can see function implementation and output format in 
evaluation_tasks -> eval_utils.py . 

If you only want the embedding of the graph, you can stop here. If you also want to apply link prediction or vertex classification task you should continue.
Line 107: export_time - Export a csv file with running times of each method according to the initial core size.
Lines 123-130- Link prediction task: A csv file of non edges is needed (as explained above), you can see comments in the code. For more details you can go to
evaluation_tasks -> link_prediction.py .
Lines 132-136- Vertex classification task: You can see comments in the code. For more details you can go to evaluation_tasks -> node_classification.py .
"""

from link_prediction import *
from node_classification import *
from static_embeddings import *
import csv

# initialize important variables / parameters
DATASET = {"name": "DBLP", "initial_size": [100, 1000], "dim": 128, "is_weighted": False, "choose": "degrees",
           "regu_val": 0, "weighted_reg": False, "s_a": True, "epsilon": 0.1,
           "label_file": os.path.join("..", "labels", "dblp_tags.txt")}

# Example for .mat
# DATASET = {"name": "Flickr", "initial_size": [1000], "dim": 128, "is_weighted": False, "choose": "degrees",
#            "regu_val": 0, "weighted_reg": False, "s_a": False, "epsilon": 0.01,
#            "label_file": os.path.join("..", "datasets", "Flickr.mat")}

datasets_path_ = os.path.join("..", "datasets")

# where to save the embeddings
if DATASET["choose"] == "degrees":
    embeddings_path_ = os.path.join("..", "embeddings_degrees")
else:
    embeddings_path_ = os.path.join("..", "embeddings_k_core")
    
# Our suggested embedding method
methods_ = ["OGRE"]
# state-of-the-art embedding methods
initial_methods_ = ["node2vec"]

# Parameters duct for state-of-the-art embedding methods
params_dict_ = {"node2vec": {"dimension": DATASET["dim"], "walk_length": 80, "num_walks": 16, "workers": 2},
                "GF": {"dimension": DATASET["dim"], "eta": 0.1, "regularization": 0.1, "max_iter": 3000,
                       "print_step": 100}, "HOPE": {"dimension": 128, "beta": 0.1},
                "GCN": {"dimension": DATASET["dim"], "epochs": 150, "lr": 0.01, "weight_decay": 5e-4, "hidden": 200,
                        "dropout": 0}}

# if you want to save the embeddings as npy file- save_=True
save_ = True

# calculate dict of embeddings
z, G, initial_size, list_initial_proj_nodes = calculate_static_embeddings(datasets_path_, embeddings_path_, DATASET,
                                                                          methods_, initial_methods_, params_dict_,
                                                                          save_=save_)

"""
if the embeddings is all you wanted you can stop here. Otherwise, here are functions to calculate running time, and 
applying Link Prediction and Node Classification Tasks.
"""

# where to save resuts files
if DATASET["choose"] == "degrees":
    save = "files_degrees"
else:
    save = "files_k_core"

# evaluate running time
export_time(z, DATASET["name"], save)
    
if DATASET["name"] == "Yelp":
    mapping = {i: n for i,n in zip(range(G.number_of_nodes()), list(G.nodes()))}
else:
     mapping=None
        
DATASET["initial_size"] = initial_size
print(initial_size)

# Link prediction Task
n = G.number_of_nodes()
non_edges_file = "non_edges_{}.csv".format(DATASET["name"])  # non edges file
# number_true_false: Number of true and false edges, number choose: How many times to choose true and false edges
params_lp_dict = {"number_true_false": 10000, "rounds": 10, "test_ratio": [0.2, 0.3, 0.5], "number_choose": 10}
dict_lp = final_link_prediction(z, params_lp_dict, non_edges_file)
export_results_lp_nc_all(n, save, z, dict_lp, DATASET["initial_size"], DATASET["name"], "Link Prediction")
print("finish link prediction")

# Node Classification Task
params_nc_dict = {"rounds": 10, "test_ratio": [0.5, 0.9]}
# for multi-label node classification add multi=True
dict_nc = final_node_classification(DATASET["name"], z, params_nc_dict, DATASET, mapping=mapping, multi=False)
export_results_lp_nc_all(n, save, z, dict_nc, DATASET["initial_size"], DATASET["name"], "Node Classification")
print("finish node classification")
