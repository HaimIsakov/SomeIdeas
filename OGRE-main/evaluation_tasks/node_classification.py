"""
Node classification Task For Evaluation
"""


try: import cPickle as pickle
except: import pickle
from sklearn.multioutput import MultiOutputClassifier
from sklearn import model_selection as sk_ms
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse
import scipy
from state_of_the_art_embedding import *


"""
Code for the node classification task as explained in GEM article. Node classification part of the code belongs to GEM
[https://github.com/palash1992/GEM]. Multi-label node classification part of the code belongs to NRL [https://github.com/PriyeshV/NRL_Benchmark].
Notice multi-label node classification may take at least 10-15 minutes for a single run.
"""

"""
Multi-label node classification
"""

def predict_top_k(classifier, X, top_k_list):
    print("predicting top k")
    assert X.shape[0] == len(top_k_list)
    probs = np.asarray(classifier.predict_proba(X))
    all_labels = []
    for i, k in enumerate(top_k_list):
        probs_ = probs[i, :]
        try:
            labels = classifier.classes_[probs_.argsort()[-k:]].tolist()
        except AttributeError:  # for eigenpro
            labels = probs_.argsort()[-k:].tolist()
        all_labels.append(labels)
    print("done predicting")
    return all_labels


def get_classifier_performance(classifer, X_test, y_test, multi_label_binarizer):
    print("check performance")
    top_k_list_test = [len(l) for l in y_test]
    y_test_pred = predict_top_k(classifer, X_test, top_k_list_test)
    
    y_test_transformed = multi_label_binarizer.transform(y_test)
    y_test_pred_transformed = multi_label_binarizer.transform(y_test_pred)
    
    micro = f1_score(y_test_transformed, y_test_pred_transformed, average="micro")
    macro = f1_score(y_test_transformed, y_test_pred_transformed, average="macro")
    acc = accuracy_score(y_test_transformed, y_test_pred_transformed)
    print(micro, macro, acc)

    return micro, macro, acc, 0
    

def sparse_tocoo(Y):
    temp_y_labels = sparse.csr_matrix(Y)
    y_labels = [[] for x in range(temp_y_labels.shape[0])]
    cy =  temp_y_labels.tocoo()
    for i, j in zip(cy.row, cy.col):
      y_labels[i].append(j)
    assert sum(len(l) for l in y_labels) == temp_y_labels.nnz
    return y_labels
      

def multi_label_logistic_regression(X, Y, test_ratio):
    number_of_labels = Y.shape[1]
    multi_label_binarizer = MultiLabelBinarizer(range(number_of_labels))
    y_fitted = multi_label_binarizer.fit(Y)
    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(X, Y, test_size=test_ratio)
    lf_classifer = oneVr(lr(solver='lbfgs', max_iter=1000), n_jobs=60)
    
    parameters = {"estimator__penalty" : ["l2"], "estimator__C": [0.001, 0.01, 0.1, 1, 10, 100]}
    lf_classifer = GridSearchCV(lf_classifer, param_grid=parameters, cv=5, scoring='f1_micro', n_jobs=1, verbose=0, pre_dispatch=1)
    
    print("done grid search")
    
    lf_classifer.fit(X_train, Y_train)
    lf_classifer = lf_classifer.best_estimator_
    y_test = sparse_tocoo(Y_test)
    
    micro, macro, acc, auc = get_classifier_performance(lf_classifer, X_test, y_test, multi_label_binarizer)
    return micro, macro, acc, auc

"""
Regular node classification
"""


class TopKRanker(oneVr):
    """
    Linear regression with one-vs-rest classifier
    """
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        prediction = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
            for label in labels:
                prediction[i, label] = 1
        return prediction


def evaluateNodeClassification(X, Y, test_ratio):
    """
    Predictions of nodes' labels.
    :param X: The features' graph- the embeddings from node2vec
    :param Y: The nodes' labels
    :param test_ratio: To determine how to split the data into train and test
    :return: Scores- F1-macro, F1-micro and accuracy.
    """
    number_of_labels = Y.shape[1]
    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(
        X,
        Y,
        test_size=test_ratio
    )
    index = []
    try:
        top_k_list = list(Y_test.toarray().sum(axis=1))
    except:
        top_k_list = list(Y_test.sum(axis=1))
    classif2 = TopKRanker(lr(solver='lbfgs', max_iter=350000))
    classif2.fit(X_train, Y_train)
    prediction = classif2.predict(X_test, top_k_list)
    accuracy = accuracy_score(Y_test, prediction)
    micro = f1_score(Y_test, prediction, average='micro')
    macro = f1_score(Y_test, prediction, average='macro')
    auc = roc_auc_score(Y_test, prediction)
    return micro, macro, accuracy, auc


"""
Code for all
"""


def expNC(X, Y, test_ratio_arr, rounds, multi=False):
    """
    The final node classification task as explained in our git.
    :param X: The features' graph- the embeddings from node2vec
    :param Y: The nodes' labels
    :param test_ratio_arr: To determine how to split the data into train and test. This an array
                with multiple options of how to split.
    :param rounds: How many times we're doing the mission. Scores will be the average
    :param multi: True for multi-label classification, else False
    :return: Scores for all splits and all splits- F1-micro, F1-macro and accuracy.
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
            if not multi:
                micro_round[i], macro_round[i], acc_round[i], auc_round[i] = evaluateNodeClassification(X, Y, test_ratio)
            else:
                micro_round[i], macro_round[i], acc_round[i], auc_round[i] = multi_label_logistic_regression(X, Y, test_ratio)

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
    :param score: F1-micro / F1-macro / Accuracy
    :param rounds: How many times the experiment has been applied for every split.
    :return: Average score for every split
    """
    all_avg_scores = []
    for i in range(score.shape[1]):
        avg_score = (np.sum(score[:, i])) / rounds
        all_avg_scores.append(avg_score)
    return all_avg_scores


def calculate_all_avg_scores_for_all(micro, macro, acc, auc, rounds):
    """
    For all scores calculate the average score for every split. The function returns list for every
    score type- 1 for cheap node2vec and 2 for regular node2vec.
    """
    all_avg_micro = calculate_avg_score(micro, rounds)
    all_avg_macro = calculate_avg_score(macro, rounds)
    all_avg_acc = calculate_avg_score(acc, rounds)
    all_avg_auc = calculate_avg_score(auc, rounds)
    return all_avg_micro, all_avg_macro, all_avg_acc, all_avg_auc


def read_labels(name, file_tags, dict_proj, mapping=None):
    """
    Read the labels file and return the labels as a matrix. Matrix is from size number of samples by number
    of labels, where C[i,j]==1 if node i has label j, else 0.
    :param file_tags: a file with labels for every node
    :return: matrix as explained above
    """
    if name == "Yelp":
        Y, dict_proj = read_yelp_labels(file_tags, mapping, dict_proj)
    elif name == "Flickr" or name == "Youtube":
        Y, dict_proj = read_mat_labels(file_tags, dict_proj) 
    else:
        if name == "Reddit":
            f = open(file_tags, 'r')
            labels = {}
            for line in f:
                name = line.split(" ")[0]
                label = int(line.split(" ")[1].split("\n")[0])
                labels.update({name: label})
            f.close()
        else:
            c = np.loadtxt(file_tags).astype(int)
            if name == "Pubmed":
                labels = {str(x): int(y - 1) for (x, y) in c}
            else:
                labels = {str(x): int(y) for (x, y) in c}
        keys = list(dict_proj.keys())
        values = list(labels.values())
        values = list(dict.fromkeys(values))
        values.sort()
        number_of_labels = values[-1] + 1
        Y = np.zeros((len(dict_proj), number_of_labels))
        for i in range(len(keys)):
            key = keys[i]
            tag = labels[str(key)]
            for j in range(number_of_labels):
                if j == tag:
                    Y[i, j] = 1
        for k in range(number_of_labels):
            if np.all((Y[:, k] == 0), axis=0):
                Y = np.delete(Y, k, 1)
    return Y, dict_proj


def read_yelp_labels(file_tags, mapping, dict_proj):
    """
    Read labels of yelp dataset
    """
    X = np.loadtxt(file_tags)
    Y = np.int_(X)
    number_of_labels = Y.shape[1]
    for k in range(number_of_labels):
        if np.all((Y[:, k] == 0), axis=0):
            Y = np.delete(Y, k, 1)
    not_here = len(dict_proj) - Y.shape[0]
    for n in range(not_here):
        del dict_proj[mapping[n]]
    return Y, dict_proj


def read_mat_labels(file_tags, dict_proj):
    features_struct = scipy.io.loadmat(file_tags)
    labels = scipy.sparse.csr_matrix(features_struct["group"])
    a = scipy.sparse.csr_matrix.toarray(labels)
    return a, dict_proj


def our_embedding_method(dict_proj, dim):
    """
    Run cheap node2vec and make it a features matrix- matrix of size number of sample by number of embedding
    dimension, where the i_th row of X is its projection from cheap node2vec.
    :param dict_proj: A dictionary with keys==nodes in projection and values==projection
    :return: a matrix as explained above
    """
    X = np.zeros((len(dict_proj), dim))
    keys = list(dict_proj.keys())
    for i in range(len(keys)):
        X[i, :] = dict_proj[keys[i]]
    return X


def nc_mission(name, key, z, ratio_arr, label_file, dim, rounds, mapping=None, multi=False):
    """
    Node Classification Task where one wants the scores as a function of size of the initial embedding. Notice test
    ratio must be fixed. The variable that changes here is the size of the initial embedding. For more  explanation,
    see our pdf file attached in out git.
    :param The applied embedding method
    :param z: Embedding dictionary of the given graph (with all types of our methods, no state-of-the-art))
    :param ratio_arr: Test ratio
    :param label_file: File with labels of the graph. For true format see "results_all_datasets.py" file.
    :param dim: Dimension of the embedding space
    :param rounds: How many time to repeat the task for evaluation
    :param multi: True for multi-label classification, else False
    :return: Scores of node classification task for each dataset- Micro-F1, Macro-F1, Accuracy and AUC. They return as
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
            Y, dict_proj = read_labels(name, label_file, list_dict_projections[j], mapping)
            X = our_embedding_method(dict_proj, dim)
            micro, macro, acc, auc = expNC(X, Y, [r], rounds, multi=multi)
            avg_micro, avg_macro, avg_acc, avg_auc = calculate_all_avg_scores_for_all(micro, macro, acc, auc, rounds)
            print(avg_micro)
            print(avg_macro)
            print(avg_acc)
            print(avg_auc)
            all_micro.append(avg_micro[0])
            all_macro.append(avg_macro[0])
            all_acc.append(avg_acc[0])
            all_auc.append(avg_auc[0])
        dict_initial.update({r: [all_micro, all_macro, all_acc, all_auc]})
    return dict_initial


def final_node_classification(name, dict_all_embeddings, params_nc, dict_dataset, mapping=None, multi=False):
    """
    Node Classification Task
    :param dict_all_embeddings: Dictionary with all dict embeddings for all applied embedding method
    :param params_nc: Parameters for node classification task
    :param multi: True for multi-label classification, else False
    :return: Dict where keys are applied methods and keys are dicts of scores for each test ratio.
    """
    dict_nc_mission = {}

    ratio_arr = params_nc["test_ratio"]
    rounds = params_nc["rounds"]

    keys = list(dict_all_embeddings.keys())

    for key in keys:
        label_file = dict_dataset["label_file"]
        d = dict_dataset["dim"]
        dict_initial = nc_mission(name, key, dict_all_embeddings, ratio_arr, label_file, d, rounds, mapping=mapping, multi=multi)
        dict_nc_mission.update({key: dict_initial})
    return dict_nc_mission
