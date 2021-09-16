import json
import os

import numpy as np


def create_params_file(file):
    params_dict = {}
    with open(file, "r") as f:
        file_name = file.split("val_")[0].replace("A", "")[:-1]
        count = 0
        for line in f:
            if line == "\n":
                count += 1
            elif count == 2:
                split_line = [word.replace("\n", "") for word in line.split(",")]
                parameter_name = split_line[0]
                parameter_value = split_line[1]
                params_dict[parameter_name] = parameter_value
    params_dict = change_type(params_dict)
    return file_name, params_dict

def change_type(params_dict):
    # int params
    params_dict["layer_1"] = int(params_dict["layer_1"])
    params_dict["preweight"] = int(params_dict["preweight"])
    params_dict["batch_size"] = int(params_dict["batch_size"])
    params_dict["layer_2"] = int(params_dict["layer_2"])
    params_dict["epochs"] = int(params_dict["epochs"])

    # float64 params
    params_dict["test_frac"] = np.float64(params_dict["test_frac"])
    params_dict["regularization"] = np.float64(params_dict["regularization"])
    params_dict["train_frac"] = np.float64(params_dict["train_frac"])
    params_dict["dropout"] = np.float64(params_dict["dropout"])
    params_dict["learning_rate"] = np.float64(params_dict["learning_rate"])
    return params_dict


if __name__ == '__main__':
    directory = "nni_best_results_and_models"
    # files_list = sorted(os.listdir("."))
    files_list = ["IBD_Chrone_just_values_val_mean_0.943_test_mean_0.915.csv",
                  "cirrhosis_just_values_val_mean_0.950_test_mean_0.876.csv"]
    for file in files_list:
        try:
            file_name, params_dict = create_params_file(file)
            with open(os.path.join("params_file_1_gcn", file_name + "_params" + ".json"), 'w') as fp:
                json.dump(params_dict, fp)
        except:
            raise
