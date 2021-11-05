"""
Plot utils for plotting some important graphs.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import csv


# for plots
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams["font.family"] = "Times New Roman"


def read_results(name, save, mission, initial_methods_, ratio, mapping):
    """
    Read the results for plots
    """
    dict_mission = {}
    dict_times = {}
    initial_size = []
    for m in initial_methods_:
        dict_times.update({m: {}})
    with open(os.path.join("..", save, "{} {}.csv".format(name, mission))) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            try:
                a = row[1]
            except:
                continue
            if a == "embed algo":
                continue
            else:
                if row[0] != "":
                    size = int(row[0])
                    if size not in initial_size:
                        initial_size.append(size)
                initial_method = row[1]
                our_method = row[2]
                if our_method != "":
                    our_method = mapping[our_method]
                    combination = initial_method + " + " + our_method
                else:
                    combination = initial_method
                test_ratio = float(row[3])
                micro = float(row[4].split("+")[0])
                macro = float(row[5].split("+")[0])
                auc = float(row[6].split("+")[0])
                if mission != "Node Classification":
                    # time = float(row[7])
                    time = 0
                else:
                    time = 0
                if dict_mission.get(combination) is None:
                    dict_mission.update({combination: {test_ratio: [[micro], [macro], [0], [auc]]}})
                else:
                    if dict_mission[combination].get(test_ratio) is None:
                        dict_mission[combination].update({test_ratio: [[micro], [macro], [0], [auc]]})
                    else:
                        dict_mission[combination][test_ratio][0].append(micro)
                        dict_mission[combination][test_ratio][1].append(macro)
                        dict_mission[combination][test_ratio][3].append(auc)
                if test_ratio == ratio:
                    for m in initial_methods_:
                        if m in combination:
                            key = our_method
                            if our_method == "":
                                key = initial_method
                            if dict_times[m].get(key) is None:
                                dict_times[m].update({key: [time]})
                            else:
                                dict_times[m][key].append(time)
    if dict_mission.get("HOPE + LGF") is not None:
        del dict_mission["HOPE + LGF"]
        del dict_mission["node2vec + LGF"]
    keys_ours = []
    keys_state_of_the_art = []
    keys = list(dict_mission.keys())
    for key in keys:
        if " " in key:
            keys_ours.append(key)
        else:
            keys_state_of_the_art.append(key)
    if mission == "Link Prediction":
        return dict_mission, dict_times, initial_size, keys_ours, keys_state_of_the_art
    else:
        return dict_mission


def choose_max_initial(dict_mission, keys_ours_, ratio_arr, initial_arr, score):
    """
    To plot the graphs choose the initial embedding size that has given the best results for each method (ours).
    """
    dict_number_initial = {}
    for key in keys_ours_:
        dict_initial = dict_mission[key]
        dict_ratio = {}
        for r in ratio_arr:
            max_score = 0
            index = -1
            for i in range(len(initial_arr)):
                if score == "Micro-F1":
                    score_1 = dict_initial[r][0][i]
                elif score == "Macro-F1":
                    score_1 = dict_initial[r][1][i]
                else:
                    score_1 = dict_initial[r][3][i]
                if score_1 > max_score:
                    max_score = score_1
                    index = i
            dict_ratio.update({r: index})
        dict_number_initial.update({key: dict_ratio})
    return dict_number_initial


def all_test_by_one_chosen_initial(dict_mission, dict_number_initial, keys_state_of_the_art_, ratio_arr, score):
    """
    Helper function to plot graph of scores as a function of test ratios values.
    """
    dict_test_score = {}
    keys = list(dict_mission.keys())
    for key in keys:
        if key not in keys_state_of_the_art_:
            dict_ratio = dict_number_initial[key]
        all_scores = []
        dict_initial = dict_mission[key]
        for r in ratio_arr:
            if key in keys_state_of_the_art_:
                index = 0
            else:
                index = dict_ratio[r]
            if score == "Micro-F1":
                score_1 = dict_initial[r][0][index]
            elif score == "Macro-F1":
                score_1 = dict_initial[r][1][index]
            else:
                score_1 = dict_initial[r][3][index]
            all_scores.append(score_1)
        dict_test_score.update({key: all_scores})
    return dict_test_score


def all_test_by_one_initial(dict_mission, keys_state_of_the_art_, ratio_arr, number_initial, initial_arr, score):
    """
    Create a dictionary of scores for each method where there are several test ratios and a chosen size of initial
    embedding.
    :param dict_mission: Dictionary of the mission
    :param keys_state_of_the_art_: state-of-the-art methods
    :param ratio_arr: List of test ratios
    :param number_initial: Chosen size of initial embedding
    :param initial_arr: Array of different sizes of initial embedding
    :param score: Chosen score
    :return: A dictionary: key == name of method , value == list of scores (of the given score) for each test ratio
                (size of initial embedding is fixed)
    """
    dict_test_score = {}
    keys = list(dict_mission.keys())
    # if number is not valid will take the first number in the initial list
    index = 0
    for i in range(len(initial_arr)):
        if initial_arr[i] == number_initial:
            index = i
            break
    for key in keys:
        all_scores = []
        if key in keys_state_of_the_art_:
            index = 0
        dict_initial = dict_mission[key]
        for r in ratio_arr:
            if score == "Micro-F1":
                score = dict_initial[r][0][index]
            elif score == "Macro-F1":
                score = dict_initial[r][1][index]
            else:
                score = dict_initial[r][3][index]
            all_scores.append(score)
        dict_test_score.update({key: all_scores})
    return dict_test_score


def all_initial_by_one_test(dict_mission, keys_ours_, keys_state_of_the_art_, initial_arr, number_test, ratio_arr, score):
    """
    Create a dictionary of scores for each method where there are several sizes of initial embedding and a chosen value
     of test ratio. Only for our methods.
    :param dict_mission: Dictionary of the mission
    :param keys_ours_: Names of our embedding methods only
    :param initial_arr: Array of sizes of initial embedding
    :param number_test: Chosen value of test ratio
    :param ratio_arr: Array of test ratios
    :param score: Chosen score
    :return: A dictionary: key == name of method , value == list of scores (of the given score) for each size of initial
                embedding (test ratio is fixed)
    """
    dict_initial_score = {}
    # if test ratio is not valid
    if number_test not in ratio_arr:
        number_test = ratio_arr[0]
    for key in keys_ours_:
        all_scores = []
        dict_initial = dict_mission[key]
        for l in range(len(initial_arr)):
            if score == "Micro-F1":
                score_1 = dict_initial[number_test][0][l]
            elif score == "Macro-F1":
                score_1 = dict_initial[number_test][1][l]
            else:
                score_1 = dict_initial[number_test][3][l]
            all_scores.append(score_1)
        for key_2 in keys_state_of_the_art_:
            if key_2 in key:
                if score == "Micro-F1":
                    score_1 = dict_mission[key_2][number_test][0][0]
                elif score == "Macro-F1":
                    score_1 = dict_mission[key_2][number_test][1][0]
                else:
                    score_1 = dict_mission[key_2][number_test][3][0]
                all_scores.append(score_1)
        dict_initial_score.update({key: all_scores})

    return dict_initial_score


def plot_test_vs_score(name, task, score, dict_mission, keys_ours , keys_state_of_the_art_, ratio_arr, initial_arr,
                       colors, i, save):
    """
    Plot graph of a given score of a given task as a function of test ratio values, for all applied methods.
    """
    keys = keys_ours + keys_state_of_the_art_
    if name == "Yelp" or name == "Reddit":
        keys = keys_ours
    dict_number_initial = choose_max_initial(dict_mission, keys_ours, ratio_arr, initial_arr, score)
    #dict_scores = all_test_by_one_initial(dict_mission, keys_state_of_the_art_, ratio_arr, number_initial, initial_arr,
     #                                    score)
    dict_scores = all_test_by_one_chosen_initial(dict_mission, dict_number_initial, keys_state_of_the_art_, ratio_arr, score)
    plt.figure(i, figsize=(7, 6))
    for j in range(len(keys)):
        if "DOGRE" in keys[j]:
            marker = '>'
            markersize = 8
            linestyle = 'solid'
        elif "WOGRE" in keys[j]:
            marker = '*'
            markersize = 10
            linestyle = 'dotted'
        elif "LGF" in keys[j]:
            marker = 'o'
            markersize = 8
            linestyle = (0, (3, 1, 1, 1))
        elif "OGRE" in keys[j]:
            marker = 'D'
            markersize = 6
            linestyle = 'dashed'
        else:
            marker = 'x'
            linestyle = (0, (3, 5, 1, 5))
        plt.plot(ratio_arr, dict_scores[keys[j]], marker=marker, linestyle=linestyle, markersize=markersize, color=colors[keys[j]])
    if task == "Node Classification":
        if score == "AUC":
            bottom = 0
            top = 1
        elif score == "Macro-F1":
            bottom = 0
            top = 1
        else:
            bottom = 0
            top = 1
    else:
        if score == "AUC":
            bottom = 0
            top = 1
        elif score == "Macro-F1":
            bottom = 0
            top = 1
        else:
            bottom = 0
            top = 1
    plt.ylim(bottom=bottom, top=top)
    plt.legend(keys, loc='best', ncol=3, fontsize='medium')
    plt.title("{} Dataset \n {} Task - {} Score".format(name, task, score))
    plt.xlabel("Test ratio")
    plt.ylabel("{}".format(score))
    plt.tight_layout()
    plt.savefig(os.path.join("..", save, "{} {} {}.png".format(name, task, score)))


def plot_initial_vs_score(name, task, score, dict_mission, keys_ours_, keys_state_of_the_art_, initial_arr, number_test, ratio_arr, n,
                       colors, i, save):
    """
    Plot graph of a given score of a given task as a function of sizes of initial embedding, for all applied methods.
    """
    if name == "Yelp" or name == "Reddit":
        keys_state_of_the_art_ = []
    dict_scores = all_initial_by_one_test(dict_mission, keys_ours_, keys_state_of_the_art_, initial_arr, number_test, ratio_arr,
                                          score)
    plt.figure(i, figsize=(7, 6))
    for j in range(len(keys_ours_)):
        if "DOGRE" in keys_ours_[j]:
            marker = '>'
            markersize = 8
            linestyle = 'solid'
        elif "WOGRE" in keys_ours_[j]:
            marker = '*'
            markersize = 10
            linestyle = 'dotted'
        elif "LGF" in keys_ours_[j]:
            marker = 'o'
            markersize = 8
            linestyle = (0, (3, 1, 1, 1))
        elif "OGRE" in keys_ours_[j]:
            marker = 'D'
            markersize = 6
            linestyle = 'dashed'
        else:
            marker = 'x'
            linestyle = (0, (3, 5, 1, 5))
        plt.xscale("log")
        plt.plot(n, dict_scores[keys_ours_[j]], marker=marker, linestyle=linestyle, markersize=markersize, color=colors[keys_ours_[j]])
    if task == "Node Classification":
        if score == "AUC":
            bottom = 0
            top = 1
        elif score == "Micro-F1":
            bottom = 0
            top = 1
        else:
            bottom = 0
            top = 1
    else:
        if score == "AUC":
            bottom = 0
            top = 1
        elif score == "Micro-F1":
            bottom = 0
            top = 1
        else:
            bottom = 0
            top = 1
    plt.legend(keys_ours_, loc='best', ncol=2, fontsize='medium')
    plt.ylim(bottom=bottom, top=top)
    plt.title("{} Dataset \n {} Task - {} Score".format(name, task, score))
    plt.xlabel("Size of initial embedding")
    plt.ylabel("{}".format(score))
    plt.tight_layout()
    plt.savefig(os.path.join("..", save, "{} {} {} initial.png".format(name, task, score)))


def plot_test_vs_score_all(name, task, dict_mission, keys_ours, keys_state_of_the_art_, ratio_arr, initial_arr,
                           colors, i, save):
    scores = ["Micro-F1", "Macro-F1", "AUC"]
    for s in scores:
        plot_test_vs_score(name, task, s, dict_mission, keys_ours, keys_state_of_the_art_, ratio_arr, initial_arr,
                           colors, i, save)
        i += 1


def plot_initial_vs_score_all(name, task, dict_mission, keys_ours_, keys_state_of_the_art_, initial_arr, number_test, ratio_arr, n,
                           colors, i, save):
    if name == "Yelp" or name == "Reddit":
        del n[-1]
    scores = ["Micro-F1", "Macro-F1", "AUC"]
    for s in scores:
        plot_initial_vs_score(name, task, s, dict_mission, keys_ours_, keys_state_of_the_art_, initial_arr, number_test, ratio_arr, n,
                           colors, i, save)
        i += 1


def read_times_file(name, save, initial_methods_, mapping):
    """
    Read times files
    """
    dict_times = {}
    for m in initial_methods_:
        dict_times.update({m: {}})
    with open(os.path.join("..", save, "{} times_1.csv".format(name))) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) == 0:
                continue
            elif row[1] == "embed algo":
                continue
            else:
                initial_method = row[1]
                our_method = row[2]
                if our_method != "":
                    our_method = mapping[our_method]
                    combination = initial_method + " + " + our_method
                else:
                    combination = initial_method
                time = float(row[3])
                for m in initial_methods_:
                    if m in combination:
                        key = our_method
                        if our_method == "":
                            key = initial_method
                        if dict_times[m].get(key) is None:
                            dict_times[m].update({key: [time]})
                        else:
                            dict_times[m][key].append(time)
    return dict_times


def plot_running_time_after_run(name, method, dict_times, initial_size, colors, i):
    """
    Plot running time for each initial method
    """
    keys = list(dict_times[method].keys())
    combinations = []
    for m in keys:
        if m != "node2vec":
            combinations.append(method + " + " + m)
        else:
            combinations.append(m)
    if name != "Yelp":
        if name != "Reddit":
            t = dict_times[method][keys[-1]][0]
    plt.figure(i, figsize=(7, 6))
    for j in range(len(keys)):
        if len(dict_times[method][keys[j]]) > 1:
            times = dict_times[method][keys[j]]
            if name != "Yelp":
                if name != "Reddit":
                    times.append(t)
            if "DOGRE" in keys[j]:
                marker = '>'
                markersize = 8
                linestyle = 'solid'
            elif "WOGRE" in keys[j]:
                marker = '*'
                markersize = 10
                linestyle = 'dotted'
            elif "LGF" in keys[j]:
                marker = 'o'
                markersize = 8
                linestyle = (0, (3, 1, 1, 1))
            elif "OGRE" in keys[j]:
                marker = 'D'
                markersize = 6
                linestyle = 'dashed'
            else:
                marker = 'x'
                linestyle = (0, (3, 5, 1, 5))
            plt.loglog(initial_size, times, marker=marker, linestyle=linestyle, markersize=markersize, color=colors[combinations[j]])
    plt.legend(keys, loc='best', ncol=2)
    plt.title("{} Dataset - {}\n Running Time VS Size of Initial Embedding".format(name, method))
    plt.xlabel("Size of initial embedding")
    plt.ylabel("Running time [seconds]")
    plt.tight_layout()
    plt.savefig(os.path.join("..", "plots", "{} {} running time vs initial.png".format(name, method)))


def plot_running_time_after_run_all(dict_dataset, dict_times, colors, n, i):
    """
    Plot running time for all initial methods
    """
    initial = dict_dataset["initial_embedding_size"]
    name = dict_dataset["name"]
    if name != "Yelp":
        if name != "Reddit":
            initial.append(n)
    keys = list(dict_times.keys())
    for key in keys:
        plot_running_time_after_run(name, key, dict_times, initial, colors, i)
        i += 1
    if name != "Yelp":
        if name != "Reddit":
            initial.remove(n)
