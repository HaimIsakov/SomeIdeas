# OGRE - Online Two-Stages Embedding Method For Large Graphs

**Contact**

******@gmail.com

## Overview
OGRE and its variants are fast online two-stages graph embedding algorithms for large graphs. The accuracy of existing embedding, as defined by auxiliary tests, is maximal for a core of high degree vertices. We propose to embed only this core using existing methods (such as node2vec, HOPE, GF, GCN), and then update online the remaining vertices, based on the position of their already embedded neighbors. The position of each new vertex is a combination of its first and second neighbors’ positions. We present three versions of this heuristic:

1. OGRE - a weighted combination approach which assumes an undirected graph, or the undirected graph underlying a directed graph. The position of a new vertex that is inserted to the embedding is calculated by the average embedding of its first and second order neighbours, with epsilon as a hyperparamter representing the importance of the second order neighbours.
2. DOGRE - a directed regression which assumes a directed graph. The position of a new vertex that is inserted to the embedding is calculated by the average embedding of its first and second order neighbours, where now they have directions - in, out, in-in, in-out, out-in, out-out, and the importance of each of them is determined by the regression weights.
3. WOGRE - a directed weighted regression, very similar to DOGRE. The difference is by the calculation of the parameters, where here we use a little different combination, therefore different regression results. 

## About This Repo
This repo contains source code of our three suggested embedding algorithms for large graphs.

### The Directories:
- `datasets` - Examples for datasets files
- `labels` - Examples for labels files
- `embeddings_degrees` - Where to save OGRE/DOGRE/WOGRE embeddings in .npy format
- `embeddings_state_of_the_art` - Where to save state-of-the-art embeddings in .npy format
- `evaluation_tasks` - Implementation of vetrex classification and link prediction tasks + main file to calculate the embedding and evaluate performance (`calculate_static_embeddings.py`).
- `files_degrees` - Where to save the results
- `our_embeddings_methods` - Implementations of our suggested embedding methods - OGRE/DOGRE/WOGRE.
- `plots` - Examples plots
- `state_of_the_art` - State-of-the-art embedding algorithms implementations, currently node2vec/GF/HOPE/GCN.

### Dependencies:

- python >=3.6.8
- numpy >= 1.18.0
- scikit-learn >= 0.22.1
- heapq 
- node2vec==0.3.2
- networkx==1.11
- scipy >= 1.41
- pytorch==1.7.0
- matplotlib==3.1.3
- pandas >= 1.0.5

### Cloning this repo

If this repository is cloned as a pycharm project, one needs to make the following directories as sources directories: `datasets`, `evaluation_tasks`, `labels`, `our_embeddings_methods`, `state_of_the_art`.

### Datasets:
- DBLP
- Pubmed
- Yelp
- Reddit
- Flickr
- Youtube

`datasets` directory consists of several small datasets. You can find the larger ones in [This Google Drive link](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz), taken from [GraphSaint public github repository](https://github.com/GraphSAINT/GraphSAINT) and from [NRL Benchmark public github repository](https://github.com/PriyeshV/NRL_Benchmark)
(go to `link prediction` or `node classification` directories, there you will find links to datasets in .mat format). Once you have your datasets, place them in the `datasets` direcotiry.

Notice you will have to adjust them to our files format (will be further explained) or provide a data loader function in order to produce the networkx graph. For .mat files, see how Youtube and Flickr datasets are loaded. You can add the appropriate condition to the function "load_data" in `evaluation_tasks` -> `eval_utils.py`. Note that when having .mat file, it has both the edges and labels. To see an example for a use go to "load_data" in `evaluation_tasks` -> `eval_utils.py` and to `evaluation_tasks` -> `calculate_static_embeddings.py`.

### What files should you have in order to embed your graph?
- In order to calculate the embedding, you first must have an edge list file:
"datasets/name_of_dataset.txt" - An edge list txt file. If the graph is unweighted it consists of 2 columns: source, target (with no title, source and target share an edge).
If the graph is weighted, it consists of 3 columns: source target weight. 
Example for unweighted graph: <br>
1 2 <br>
2 3 <br>
1 4 <br>
1 3 <br>
Example for weighted graph: <br>
1 2 3 <br>
1 3 0.3 <br>
1 4 4.5 <br>
2 4 0.98 <br>
You can see examples for this format in `datasets` directory.

Note that in the end one must have a networkx graph, so you can change the data loader function as you want (adjusting to your file format), but remember a networkx graph is requierd in the end.

Another format is `.mat`, as explained above. For example you can see `datasets\Flickr.mat` and see how it is loaded in `evaluation_tasks`->`eval_utils.py`->`load_data`.
- If you want to peform vertex classification task or GCN initial embedding is used, you must provide labels file: <br>
"labels/name_of_dataset_tags.txt" - A txt file which consists of 2 columns: node, label (no title). Notice all nodes must have labels! <br>
Example: <br>
1 0 <br>
2 0 <br>
3 1 <br>
4 2 <br>
You can see examples for this format in `labels` directory.

If your file is in `.mat` format, you do not need labels file (because it already has labels).
- If you want to perform link prediction task, you must provide non edges file: <br>
"evaluation_tasks/non_edges_{name_of_dataset}" - A csv file which consists of two columns: node1, node2 ; where there is no edge between them (again no title). <br>
In order to produce such file, you can go to `evaluation_tasks -> calculate_non_edges.py` , and follow the instructions there.

### How to run?
The main file is `evaluation_tasks -> calculate_static_embeddings.py`.

When you have all the files you need (depending on what you want to perform), you can run this file. These instructions are also appear in the specific file, so you can go
there and see both instructions and code.
1. First initialize DATASET parameters dict:
- name: Name of dataset (as the name of the edge list txt file) (string)
- initial_size: List of initial core sizes. (list)
- dim: Embedding dimension (int)
- is_weighted: True if the graph is weighted, else False (bool)
- choose: "degrees" if the vertices of the initial core are the ones with highest degree (as done in our experiments), else "k_core" if the vertices of the initial core are
the ones with highest k-core score. (string)
- "s_a": True if you also want to calculate state-of-the-art embeddings (node2vec/GF/HOPE/GCN), else False.
Params for OGRE:
- epsilon: Weight to the second order neighbours embedding. For more details you can go to the implementation- `our_embedding_methods -> OGRE.py` (float).
Params for DOGRE/WOGRE:
- "regu_val": Regularization value for regression, only for DOGRE/WOGRE. For more details you can go to the implementation- `our_embedding_methods -> D_W_OGRE.py` (float).
- "weighted_reg": True for weighted regression, else False.
If the initial embedding method is GCN and/or a vertex classification task is applied, a labels file is also necessary:
- "label_file": path and name (together), so it can be read directly.
2. methods_ : List of our suggested embedding methods (OGRE/DOGRE/WOGRE) with whom you want to embed the given graph. 
3. initial_methods_ : List of state-of-the-art embedding methods (node2vec/GF/HOPE/GCN) with whom the initial core will be embed.
4. params_dict_ : Parameters for state-of-the-art embeddings. These are the optimal ones (according to their papers). For more details you can go to- 
`state_of_the_art -> state_of_the_art_embedding.py`
5. save_: True if you want to save the embedding in a .npy format, else False. 

Once you have all the above, you can run "calculate_static_embeddings" function to get the embeddings as dictionaries. You can see function implementation and output format in 
`evaluation_tasks -> eval_utils.py`. <br>
If you only want the embedding of the graph, you can stop here. If you also want to apply link prediction or vertex classification task you should continue. <br>
Line 122: export_time - Export a csv file with running times of each method according to the initial core size. <br>
Lines 132-139- Link prediction task: A csv file of non edges is needed (as explained above), you can see comments in the code. For more details you can go to
`evaluation_tasks -> link_prediction.py`. <br>
Lines 141-145- Vertex classification task: You can see comments in the code. For more details you can go to `evaluation_tasks -> node_classification.py`. Notice that for a multi-label node classification task, you need to add `multi=True` in the `final_node_classification` function.

If link prediction/vertex classificaion tasks have been performed, one can also plot graphs of scores as function of size of the core, scores as function of test ratio and running time. To do that, toy can go to `evaluation_tasks -> plot_results.py`.
