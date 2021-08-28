import argparse
import pandas as pd
import os

def get_reported_results(trial_log_file):
    try:
        q = trial_log_file.readlines()
        results = [line.split("Mean")[1] for line in q[-2:]]
        mean_val, std_val = [float(i) for i in results[0].split(":")[1].replace(" ", "").replace("\n", "").split("+-")]
        mean_test, std_test = [float(i) for i in results[1].split(":")[1].replace(" ", "").replace("\n", "").split("+-")]
    except:
        mean_val, std_val, mean_test, std_test = -1, -1, -1, -1
    return mean_val, std_val, mean_test, std_test

def get_hyper_parameters_as_dict(params_file):
    f = open(params_file, "r")
    parames_dict = {}
    for line in f:
        x = line.split(",")
        x = x[2:-2]
        x = [i.replace("\'", "").replace("\"", "").replace(" ", "").replace("{", "") for i in x]
        x[0] = x[0].replace("parameters:", "")
        for i in x:
            y = i.split(":")
            try:
                y[1] = float(y[1])
            except:
                pass
            parames_dict[y[0]] = y[1]
    return parames_dict


def get_nni_results(nni_experiment_id, result_path):
    nni_experiment_path = os.path.join("nni-experiments", nni_experiment_id)
    trials_path = os.path.join(nni_experiment_path, "trials")
    nni_results_dict = {}
    for root, dirs, files in os.walk(trials_path):
        for dir in dirs:
            #print(dir)
            if dir == ".nni":
              continue
            params_file = os.path.join(root, dir, "parameter.cfg")
            nni_results_dict[dir] = get_hyper_parameters_as_dict(params_file)
            trial_log_file = open(os.path.join(root, dir, "trial.log"), 'r')
            mean_val, std_val, mean_test, std_test = get_reported_results(trial_log_file)
            nni_results_dict[dir]["mean_val_nni"] = mean_val
            nni_results_dict[dir]["std_val_nni"] = std_val
            nni_results_dict[dir]["mean_test_nni"] = mean_test
            nni_results_dict[dir]["std_test_nni"] = std_test
    results_df = pd.DataFrame.from_dict(nni_results_dict)
    results_df.T.to_csv(result_path)
    return results_df


parser = argparse.ArgumentParser(description='Export nni results menu')
parser.add_argument("--nni_id", help="nni experiment id", type=str)
parser.add_argument("--result_file_name", help="result file path", type=str)

args = parser.parse_args()
nni_experiment_id = args.nni_id
result_file_path = args.result_file_name
results_df = get_nni_results(nni_experiment_id, result_file_path)
