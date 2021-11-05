"""
Link Prediction task for evaluation. Code was mainly taken from [https://github.com/PriyeshV/NRL_Benchmark]
"""

try: import cPickle as pickle
except: import pickle
from numpy import linalg as LA
from sklearn import model_selection as sk_ms
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.linear_model import LogisticRegression as lr
import random
from eval_utils import *


def choose_true_edges(edges, K):
    """
    Randomly choose a fixed number of existing edges
    :param edges: The graph's edges
    :param K: Fixed number of edges to choose
    :return: A list of K true edges
    """
    indexes = random.sample(range(1, len(edges)), K)
    true_edges = []
    for i in indexes:
        true_edges.append(edges[i])
    return true_edges


def choose_false_edges(non_edges, K):
    """
    Randomly choose a fixed number of non-existing edges
    :param non_edges: Edges that are not in the graph
    :param K: Fixed number of edges to choose
    :return: A list of K false edges
    """
    indexes = random.sample(range(1, len(non_edges)), K)
    false_edges = []
    for i in indexes:
        false_edges.append(non_edges[i])
    return false_edges


def calculate_classifier_value(dict_projections, true_edges, false_edges, K, mapping=None):
    """
    Create X and Y for Logistic Regression Classifier.
    :param dict_projections: A dictionary of all nodes emnbeddings, where keys==nodes and values==embeddings
    :param true_edges: A list of K false edges
    :param false_edges: A list of K false edges
    :param K: Fixed number of edges to choose
    :param mapping: Only for yelp dataset, ignore otherwise
    :return: X - The feature matrix for logistic regression classifier. Its size is 2K,1 and the the i'th row is the
                norm score calculated for each edge, as explained in the attached pdf file.
            Y - The edges labels, 0 for true, 1 for false
    """
    X = np.zeros(shape=(2 * K, 1))
    Y = np.zeros(shape=(2 * K, 1))
    count = 0
    node = list(dict_projections.keys())[0]
    a = False if isinstance(node, str) is True else True

    for edge in true_edges:
        if mapping is not None:
            edge = (mapping[edge[0]], mapping[edge[1]])
        if dict_projections.get(edge[0]) is None or dict_projections.get(edge[1]) is None:
            continue
        embd1 = dict_projections[edge[0]]
        embd2 = dict_projections[edge[1]]
        norm = LA.norm(embd1 - embd2, 2)
        X[count, 0] = norm
        Y[count, 0] = int(1)
        count += 1
    for edge in false_edges:
        if a:
            edge = (int(edge[0]), int(edge[1]))
        if dict_projections.get(edge[0]) is None or dict_projections.get(edge[1]) is None:
            continue
        embd1 = dict_projections[edge[0]]
        embd2 = dict_projections[edge[1]]
        norm = LA.norm(embd1 - embd2, 2)
        X[count, 0] = norm
        Y[count, 0] = int(0)
        count += 1
    return X, Y.ravel()


def create_model(X, Y, test_ratio):
    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(X, Y, test_size=test_ratio)
    model = lr()
    parameters = {"penalty":["l2"],"C":[0.01,0.1,1]}
    model = GridSearchCV(model, param_grid=parameters, cv=2, scoring='roc_auc', n_jobs=28, verbose=0,pre_dispatch='n_jobs')
    model.fit(X_train, Y_train)
    train_prob_preds = model.predict_proba(X_train)[:,1]
    test_prob_preds = model.predict_proba(X_test)[:,1]
    del model
    train_auc = roc_auc_score(Y_train, train_prob_preds)
    test_auc = roc_auc_score(Y_test, test_prob_preds)
    micro = 0
    macro = 0 
    accuracy = 0
    return micro, macro, accuracy, test_auc


def exp_lp(X, Y, test_ratio_arr, rounds):
    """
    The final node classification task as explained in our git.
    :param X: The features' graph- norm
    :param Y: The edges labels- 0 for true, 1 for false
    :param test_ratio_arr: To determine how to split the data into train and test. This an array
                with multiple options of how to split.
    :param rounds: How many times we're doing the mission. Scores will be the average.
    :return: Scores for all splits and all splits- F1-micro, F1-macro accuracy and auc
    """
    micro = [None] * rounds
    macro = [None] * rounds
    acc = [None] * rounds
    auc = [None] * rounds

    for round_id in range(rounds):
        micro_round = [None] * len(test_ratio_arr)
        macro_round = [None] * len(test_ratio_arr)
        acc_round = [None] * len(test_ratio_arr)
        auc_round = [None] * len(test_ratio_arr)

        for i, test_ratio in enumerate(test_ratio_arr):
            micro_round[i], macro_round[i], acc_round[i], auc_round[i] = create_model(X, Y, test_ratio)

        micro[round_id] = micro_round
        macro[round_id] = macro_round
        acc[round_id] = acc_round
        auc[round_id] = auc_round

    micro = np.asarray(micro)
    macro = np.asarray(macro)
    acc = np.asarray(acc)
    auc = np.asarray(auc)

    return micro, macro, acc, auc


def calculate_avg_score(score, rounds):
    """
    Given the lists of scores for every round of every split, calculate the average score of every split.
    :param score: F1-micro / F1-macro / Accuracy / Auc
    :param rounds: How many times the experiment has been applied for each split.
    :return: Average score for every split
    """
    all_avg_scores = []
    for i in range(score.shape[1]):
        avg_score = (np.sum(score[:, i])) / rounds
        all_avg_scores.append(avg_score)
    return all_avg_scores


def calculate_all_avg_scores_lp(micro, macro, acc, auc, rounds):
    """
    For all scores calculate the average score for every split. The function returns list for every
    score type- 1 for cheap node2vec and 2 for regular node2vec.
    """
    all_avg_micro = calculate_avg_score(micro, rounds)
    all_avg_macro = calculate_avg_score(macro, rounds)
    all_avg_acc = calculate_avg_score(acc, rounds)
    all_avg_auc = calculate_avg_score(auc, rounds)
    return all_avg_micro, all_avg_macro, all_avg_acc, all_avg_auc


def initialize_scores():
    """
    Helper function to initialize the scores for link prediction mission
    """
    my_micro = [0, 0, 0, 0, 0]
    my_macro = [0, 0, 0, 0, 0]
    my_acc = [0, 0, 0, 0, 0]
    my_auc = [0, 0, 0, 0, 0]
    return my_micro, my_macro, my_acc, my_auc


def first_help_calculate_lp(score, avg_score):
    """
    Helper function for scores calculation
    """
    score = [x + y for x, y in zip(score, avg_score)]
    return score


def second_help_calculate_lp(score, number_of_sub_graphs):
    """
    Helper function for scores calculation
    """
    score = [x / number_of_sub_graphs for x in score]
    return score


def lp_mission(key, number_true_false, z, edges, non_edges, ratio_arr, rounds, number_choose):
    """
    Link prediction Task where one wants the scores as a function of size of the initial embedding. Notice test ratio
    must be fixed. The variable that changes here is the size of the initial embedding. For more  explanation, see our
    pdf file attached in out git.
    :param key: Name of the method
    :param number_true_false: Number of true (and false) edges to take
    :param z: Embedding dictionary of the given graph (with all types of our methods, no state-of-the-art)
    :param edges: List of edges of the given graph
    :param non_edges: How many sub graphs to create for evaluation
    :param ratio_arr: Test ratio
    :param rounds: How many rounds to repeat the score calculation.
    :param number_choose: Number of times to choose random edges
    :return: Scores of link prediction task for each dataset- Micro-F1, Macro-F1, Accuracy and AUC. They return as
            lists for each size of initial embedding for each method
    """
    dict_initial = {}
    for r in ratio_arr:
        all_micro = []
        all_macro = []
        all_acc = []
        all_auc = []
        if " + " in key:
            list_dict_projections = z[key].list_dicts_embedding
        else:
            list_dict_projections = [z[key][1]]
        for j in range(len(list_dict_projections)):
            my_micro, my_macro, my_acc, my_auc = initialize_scores()
            for i in range(number_choose):
                true_edges = choose_true_edges(edges, number_true_false)
                false_edges = choose_false_edges(non_edges, number_true_false)
                X, Y = calculate_classifier_value(list_dict_projections[j], true_edges, false_edges, number_true_false)
                micro, macro, acc, auc = exp_lp(X, Y, [r], rounds)
                avg_micro, avg_macro, avg_acc, avg_auc = calculate_all_avg_scores_lp(micro, macro, acc, auc, rounds)
                my_micro = first_help_calculate_lp(my_micro, avg_micro)
                my_macro = first_help_calculate_lp(my_macro, avg_macro)
                my_acc = first_help_calculate_lp(my_acc, avg_acc)
                my_auc = first_help_calculate_lp(my_auc, avg_auc)
            my_micro = second_help_calculate_lp(my_micro, number_choose)
            my_macro = second_help_calculate_lp(my_macro, number_choose)
            my_acc = second_help_calculate_lp(my_acc, number_choose)
            my_auc = second_help_calculate_lp(my_auc, number_choose)
            print(my_micro)
            print(my_macro)
            print(my_acc)
            print(my_auc)
            all_micro.append(my_micro[0])
            all_macro.append(my_macro[0])
            all_acc.append(my_acc[0])
            all_auc.append(my_auc[0])
        dict_initial.update({r: [all_micro, all_macro, all_acc, all_auc]})
    return dict_initial


def final_link_prediction(dict_all_embeddings, params_lp, file, mapping=None):
    """
    Link Prediction Task
    :param dict_all_embeddings: Dictionary with all dict embeddings for all applied embedding method
    :param params_lp: Parameters for link prediction task
    :return: Dict where keys are applied methods and keys are dicts of scores for each test ratio.
    """
    dict_lp_mission = {}

    number_true_false = params_lp["number_true_false"]
    rounds = params_lp["rounds"]
    ratio_arr = params_lp["test_ratio"]
    number_choose = params_lp["number_choose"]

    keys = list(dict_all_embeddings.keys())
    G = dict_all_embeddings[keys[0]].graph
    edges = list(G.edges())

    non_edges = []
    csvfile = open(file, 'r', newline='')
    obj = csv.reader(csvfile)
    for row in obj:
        non_edges.append((row[0], row[1]))

    for key in keys:
        dict_initial = lp_mission(key, number_true_false, dict_all_embeddings, edges, non_edges, ratio_arr, rounds,
                                  number_choose)
        dict_lp_mission.update({key: dict_initial})

    return dict_lp_mission
