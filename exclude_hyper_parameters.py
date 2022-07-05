import json
import os


def get_hyper_parameters_as_dict(params_file):
    f = open(params_file, "r")
    parames_dict = {}
    c = 0
    previous_line = ""
    for line in f:
        # if line == "\n":
        #     c += 1
        # if c == 2:
        #     break
        if line == "\n" and previous_line == "\n":
            break
        previous_line = line

    for line in f:
        x = line.split(",")
        # x = x[2:-1]
        x = [i.replace("\n", "") for i in x]
        try:
            parames_dict[x[0]] = float(x[1])
        except:
            try:
                parames_dict[x[0]] = int(x[1])
            except:
                parames_dict[x[0]] = x[1]
    return parames_dict


if __name__ == '__main__':
    # for dirpath, dirnames, filenames in os.walk(os.path.join("YoramAttention", "reported_results")):
    #     for file in filenames:
    #         params_dict = get_hyper_parameters_as_dict(os.path.join("YoramAttention","reported_results", file))
    #         file_name = file.split("_val_")[0]
    #         with open(file_name + ".json", 'w') as fp:
    #             json.dump(dict(sorted(params_dict.items())), fp)
    # params_file = "nugent_graph_and_values_val_mean_0.974_test_mean_0.971.csv"
    params_file = "nugent_yoram_attention_val_mean_0.992_test_mean_0.961.csv"

    parames_dict = get_hyper_parameters_as_dict(params_file)
    x=1
