import __init__
import csv
from our_embeddings_methods.static_embeddings import *
from state_of_the_art.state_of_the_art_embedding import *
import networkx as nx
from plots_utils import choose_max_initial, all_test_by_one_chosen_initial
import os
import pickle
import scipy
from scipy import io


def add_weights(G):
    edges = list(G.edges())
    for e in edges:
        G[e[0]][e[1]] = {"weight": 1}
    return G


def load_graph(path, name, is_weighted):
    """
    Data loader assuming the format is a text file with columns of : target source (e.g. 1 2) or target source weight
    (e.g. 1 2 0.34). If you have a different format, you may want to create your own data loader.
    :param path: The path to the edgelist file
    :param name: The name of te dataset
    :param is_weighted: True if the graph is weighted, False otherwise.
    :return: A Directed networkx graph with an attribute of "weight" for each edge.
    """
    if name == "Yelp":
        with open(os.path.join(path, "yelp_data.p"), 'rb') as f:
            G = pickle.load(f)
        G = add_weights(G)
    elif name == "Youtube" or name == "Flickr":
        inputFile = os.path.join(path, "{}.mat".format(name))
        features_struct = scipy.io.loadmat(inputFile)
        data = scipy.sparse.csr_matrix(features_struct["network"])
        G = nx.from_scipy_sparse_matrix(data)
        # no need to add weights, already has
    else:
        if is_weighted:
            G = nx.read_weighted_edgelist(os.path.join(path, name + ".txt"), create_using=nx.DiGraph(),
                                          delimiter=",")
            if G.number_of_nodes() == 0:
                G = nx.read_weighted_edgelist(os.path.join(path, name + ".txt"), create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(os.path.join(path, name + ".txt"), create_using=nx.DiGraph(), delimiter=",")
            if G.number_of_nodes() == 0:
                G = nx.read_edgelist(os.path.join(path, name + ".txt"), create_using=nx.DiGraph())
            # put weights equal to 1
            G = add_weights(G)
    return G


def calculate_static_embeddings(datasets_path, embeddings_path, dict_dataset, methods, initial_methods, params_dict,
                                from_files=False, save_=False):
    """
    Function to calculate static embedding, both by ours and state-of-the-art methods.
    :param datasets_path: Path to where the datasets are
    :param embeddings_path: Path to were the embeddings meed to be saved. Notice state-of-the-art methods are saved in
                            a different path.
    :param dict_dataset: Dict of dataset's important parameters, see example later.
    :param methods: List of state-of-the-art methods for initial embedding- "node2vec", "HOPE" or "GF"/
    :param initial_methods: List of our suggested embedding methods- "OGRE", "DOGRE", "WOGRE" or "LGF".
    :param params_dict: Dict of parameters for state-of-the-art methods
    :param from_files: False if you want to calculate the embeddings, True if you want to read them from a .npy file
                       format.
    :param save_: True ig you want to save the embeddings, else False.
    :param h: Only for Yelp dataset read the graph differently. All other times it is None (not needed).
    :return: A dictionary where keys are embedding methods that were applied and values are list of embedding dicts
             for each embedding method.
    """
    name = dict_dataset["name"]
    initial_size = dict_dataset["initial_size"]
    dim = dict_dataset["dim"]
    is_weighted = dict_dataset["is_weighted"]
    choose = dict_dataset["choose"]
    regu_val = dict_dataset["regu_val"]
    weighted_reg = dict_dataset["weighted_reg"]
    s_a = dict_dataset["s_a"]
    epsilon = dict_dataset["epsilon"]
    file_tags = dict_dataset["label_file"]

    G = load_graph(datasets_path, name, is_weighted=is_weighted)

    my_dict = {}
    state_of_the_art_dict = {}

    for i in range(len(methods)):
        for j in range(len(initial_methods)):
            print("start {} + {}".format(methods[i], initial_methods[j]))
            # calculate static embedding by given method
            SE = StaticEmbeddings(name, G, initial_size, initial_method=initial_methods[j], method=methods[i],
                                  dim=dim, choose=choose, regu_val=regu_val, weighted_reg=weighted_reg,
                                  epsilon=epsilon, file_tags=file_tags)
            if save_:
                SE.save_embedding(embeddings_path)
            key = "{} + {}".format(methods[i], initial_methods[j])
            # save the embedding in a dictionary with all embedding methods
            my_dict.update({key: SE})
            if i == 0:
                mmm = SE.initial_size
                list_initial_nodes = SE.list_initial_proj_nodes

    if s_a:
        if name != "Yelp":
            if name != "Reddit":
                for im in initial_methods:
                    X, projections, t = final(name, G, im, params_dict[im], file_tags=file_tags)
                    state_of_the_art_dict.update({im: [X, projections, t]})
                    if save_:
                        save_embedding_state_of_the_art(os.path.join("..", "embeddings_state_of_the_art"),
                                                        projections, name, im)

    z = {**my_dict, **state_of_the_art_dict}
    print(list(z.keys()))

    return z, G, mmm, list_initial_nodes


def export_time(z, name, save):
    """
    Export running times to csv file.
    :param z: Dict of lists of embeddings dicts.
    :param name: Name of the dataset
    """
    list_dicts = []
    csv_columns = ["initial size", "embed algo", "regression", "time"]
    csv_file = os.path.join("..", save, "{} times_1.csv".format(name))
    keys = list(z.keys())
    for key in keys:
        if " + " in key:
            se = z[key]
            initial_method = se.initial_method
            method = se.embedding_method
            for j in range(len(se.list_dicts_embedding)):
                data_results = {"initial size": se.initial_size[j], "embed algo": initial_method, "regression": method,
                                "time": se.times[j]}
                list_dicts.append(data_results)
        else:
            data_results = {"initial size": "", "embed algo": key, "regression": "", "time": z[key][2]}
            list_dicts.append(data_results)
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in list_dicts:
            writer.writerow(data)


def divide_to_keys(dict_all_embeddings):
    """
    Distinguish between types of embedding methods - ours and state-of-the-art
    :param dict_all_embeddings: dict of all embeddings
    :return: Names of our methods, names of state-of-the-art methods
    """
    keys_ours = []
    keys_state_of_the_art = []
    keys = list(dict_all_embeddings.keys())
    for key in keys:
        # value = dict_all_embeddings[key]
        if " + " in key:
            keys_ours.append(key)
        else:
            keys_state_of_the_art.append(key)
    return keys_ours, keys_state_of_the_art


def order_by_different_methods(dict_embeddings, initial_methods):
    """
    Create from the embedding dictionary 2 necessary dicts.
    :param dict_embeddings:
    :param initial_methods:
    :return: 2 dictionaries:
                1. dict_of_connections: key == state-of-the-art-method , value == list of our methods that key is their
                    initial embedding
                2. dict_methods: key == state-of-the-art-method , value == list of embedding dictionaries of the methods
                    that key is their initial embedding
    """
    dict_connections_methods = {}
    dict_of_methods = {}
    for m in initial_methods:
        dict_connections_methods.update({m: []})
        dict_of_methods.update({m: []})
    keys = list(dict_embeddings.keys())
    for key in keys:
        my_list = dict_embeddings[key]
        if len(my_list) > 3:
            our_method = my_list[0]
            method = my_list[1]
            dict_connections_methods[method].append(our_method)
            dict_of_methods[method].append(dict_embeddings[key])
        else:
            dict_of_methods[key].append(dict_embeddings[key])
    return dict_of_methods, dict_connections_methods


def order_results_by_different_initial_methods(dict_embeddings, initial_methods, dict_mission):
    """
    Create from the mission dictionary (containing results for each embeddings method) 1 necessary dict.
    :param dict_embeddings: Dictionary of all embeddings
    :param initial_methods: List of names of state-of-the-art embedding methods
    :param dict_mission: Dictionary of the mission, containing results (scores)
    :return: dict_of_methods: key == state-of-the-art-method , value == list of mission dictionaries of the methods
                    that key is their initial embedding, including the initial embedding itself.
    """
    keys = list(dict_embeddings.keys())
    dict_of_methods = {}
    for m in initial_methods:
        dict_of_methods.update({m: []})
    for key in keys:
        my_list = dict_embeddings[key]
        if len(my_list) > 3:
            method = my_list[1]
            dict_of_methods[method].append(dict_mission[key])
        else:
            dict_of_methods[key].append(dict_mission[key])
    return dict_of_methods


def calculate_std(r, i, dict_mission, keys, keys_ours, keys_state_of_the_art):
    """
    Calculate the variance of the results
    """
    all_micro = []
    all_macro = []
    all_auc = []
    for key in keys_ours:
        dict_initial = dict_mission[key]
        micro_f1 = dict_initial[r][0][i]
        macro_f1 = dict_initial[r][1][i]
        auc = dict_initial[r][3][i]
        all_micro.append(micro_f1)
        all_macro.append(macro_f1)
        all_auc.append(auc)
    for key in keys_state_of_the_art:
        dict_initial = dict_mission[key]
        micro_f1 = dict_initial[r][0][0]
        macro_f1 = dict_initial[r][1][0]
        auc = dict_initial[r][3][0]
        all_micro.append(micro_f1)
        all_macro.append(macro_f1)
        all_auc.append(auc)
    std_micro = str(round(np.std(all_micro), 3))
    std_macro = str(round(np.std(all_macro), 3))
    std_auc = str(round(np.std(all_auc), 3))
    return std_micro, std_macro, std_auc


def create_dicts_for_results(dict_all_embeddings, dict_mission, our_initial, n):
    """
    Create dictionary of results and more information to create useful csv files of results for a given mission
    :param dict_all_embeddings: Dictionary f embeddings
    :param dict_mission: Dictionary of the given mission
    :param our_initial: Array of different sizes of initial embedding.
    :param n: Number of nodes in the graph.
    :return:
    """
    keys_ours, keys_state_of_the_art = divide_to_keys(dict_all_embeddings)
    keys = list(dict_all_embeddings.keys())

    list_dicts = []

    for key in keys:
        if key in keys_ours:
            embd_algo = dict_all_embeddings[key][1]
            regression = dict_all_embeddings[key][0]
            initial = our_initial
        else:
            embd_algo = key
            regression = ""
            initial = [n]
            t = round(dict_all_embeddings[key][2], 3)
        dict_results_by_arr = dict_mission[key]
        ratio_arr = list(dict_results_by_arr.keys())
        for r in ratio_arr:
            all_micro = dict_results_by_arr[r][0]
            all_macro = dict_results_by_arr[r][1]
            all_auc = dict_results_by_arr[r][3]
            for i in range(len(initial)):
                std_micro, std_macro, std_auc = calculate_std(r, i, dict_mission, keys_ours, keys_state_of_the_art)
                if key in keys_ours:
                    t = round(dict_all_embeddings[key][8][i])
                initial_size = initial[i]
                test_ratio = r
                micro_f1 = float(round(all_micro[i], 3))
                macro_f1 = float(round(all_macro[i], 3))
                auc = float(round(all_auc[i], 3))
                if key in keys_state_of_the_art:
                    initial_size = ""
                dict_results = {"initial size": initial_size, "embed algo": embd_algo, "regression": regression,
                                "test": test_ratio, "micro-f1": str(micro_f1)+"+-"+std_micro,
                                "macro-f1": str(macro_f1)+"+-"+std_macro, "auc": str(auc)+"+-"+std_auc, "time": t}
                list_dicts.append(dict_results)
    return list_dicts


def export_results(n, dict_all_embeddings, dict_mission, our_initial, name, mission):
    """
    Write the results to csv file
    """
    csv_columns = ["initial size", "embed algo", "regression", "test", "micro-f1", "macro-f1", "auc", "time"]
    dict_data = create_dicts_for_results(dict_all_embeddings, dict_mission, our_initial, n)
    csv_file = os.path.join("..", "files", "{} {}.csv".format(name, mission))
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def export_results_from_files(dict_all_embeddings, dict_mission, our_initial, n):
    """
    Export results from Link Prediction / Node Classification Results files
    """
    keys = list(dict_all_embeddings.keys())
    keys_ours = []
    keys_state_of_the_art = []
    for key in keys:
        if "+" in key:
            keys_ours.append(key)
        else:
            keys_state_of_the_art.append(key)

    list_dicts = []

    for key in keys:
        if key in keys_ours:
            embd_algo = key.split(" + ")[0]
            regression = key.split(" + ")[1]
            initial = our_initial
        else:
            embd_algo = key
            regression = ""
            initial = [n]
        dict_results_by_arr = dict_mission[key]
        ratio_arr = list(dict_results_by_arr.keys())
        for r in ratio_arr:
            all_micro = dict_results_by_arr[r][0]
            all_macro = dict_results_by_arr[r][1]
            all_auc = dict_results_by_arr[r][3]
            for i in range(len(initial)):
                std_micro, std_macro, std_auc = calculate_std(r, i, dict_mission, keys_ours,
                                                              keys_state_of_the_art)
                initial_size = initial[i]
                test_ratio = r
                micro_f1 = float(round(all_micro[i], 3))
                macro_f1 = float(round(all_macro[i], 3))
                auc = float(round(all_auc[i], 3))
                if key in keys_state_of_the_art:
                    initial_size = ""
                dict_results = {"initial size": initial_size, "embed algo": embd_algo, "regression": regression,
                                "test": test_ratio, "micro-f1": str(micro_f1) + "+-" + std_micro,
                                "macro-f1": str(macro_f1) + "+-" + std_macro, "auc": str(auc) + "+-" + std_auc}
                list_dicts.append(dict_results)
    return list_dicts


"""
Export tasks without time
"""


def export_results_lp_nc(dict_all_embeddings, dict_mission, our_initial, n):
    keys = list(dict_all_embeddings.keys())
    keys_ours = []
    keys_state_of_the_art = []
    for key in keys:
        if "+" in key:
            keys_ours.append(key)
        else:
            keys_state_of_the_art.append(key)

    list_dicts = []

    for key in keys:
        if key in keys_ours:
            embd_algo = key.split(" + ")[1]
            regression = key.split(" + ")[0]
            initial = our_initial
        else:
            embd_algo = key
            regression = ""
            initial = [n]
        dict_results_by_arr = dict_mission[key]
        ratio_arr = list(dict_results_by_arr.keys())
        for r in ratio_arr:
            all_micro = dict_results_by_arr[r][0]
            all_macro = dict_results_by_arr[r][1]
            all_auc = dict_results_by_arr[r][3]
            for i in range(len(initial)):
                std_micro, std_macro, std_auc = calculate_std(r, i, dict_mission, keys, keys_ours,
                                                              keys_state_of_the_art)
                initial_size = initial[i]
                test_ratio = r
                micro_f1 = float(round(all_micro[i], 3))
                macro_f1 = float(round(all_macro[i], 3))
                auc = float(round(all_auc[i], 3))
                if key in keys_state_of_the_art:
                    initial_size = ""
                dict_results = {"initial size": initial_size, "embed algo": embd_algo, "regression": regression,
                                "test": test_ratio, "micro-f1": str(micro_f1) + "+-" + std_micro,
                                "macro-f1": str(macro_f1) + "+-" + std_macro, "auc": str(auc) + "+-" + std_auc}
                list_dicts.append(dict_results)
    return list_dicts


def export_results_lp_nc_all(n, save, dict_all_embeddings, dict_mission, our_initial, name, mission):
    csv_columns = ["initial size", "embed algo", "regression", "test", "micro-f1", "macro-f1", "auc"]
    dict_data = export_results_lp_nc(dict_all_embeddings, dict_mission, our_initial, n)
    csv_file = os.path.join("..", save, "{} {}.csv".format(name, mission))
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def export_best_results(name, mission, dict_mission, keys_ours_, keys_state_of_the_art_ ,ratio_arr, initial_arr, scores, index_ratio):
    csv_columns = ["embed algo", "regression", "micro-f1", "macro-f1", "auc"]
    list_dicts = []
    for key in keys_ours_:
        all_scores = []
        for score in scores:
            dict_number_initial = choose_max_initial(dict_mission, keys_ours_, ratio_arr, initial_arr, score)
            dict_test_score = all_test_by_one_chosen_initial(dict_mission, dict_number_initial, keys_state_of_the_art_,
                                                             ratio_arr, score)
            my_score = dict_test_score[key][index_ratio]
            all_scores.append(my_score)
        dict_results = {"embed algo": key.split(" + ")[0] , "regression": key.split(" + ")[1],
                        "micro-f1": str(all_scores[0]),
                        "macro-f1": str(all_scores[1]), "auc": str(all_scores[2])}
        list_dicts.append(dict_results)
    csv_file = os.path.join("..", "files", "{} {} best results.csv".format(name, mission))
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in list_dicts:
                writer.writerow(data)
    except IOError:
        print("I/O error")
