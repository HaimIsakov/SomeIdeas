
from os import listdir
from os.path import isfile, join
import os

import pandas as pd

# mypath = os.path.join("hla_results")
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# alleles_list = []
# for file in onlyfiles:
#     allele = file.split("_")[1]
#     alleles_list.append(allele)
#
# train_tag_file_path = os.path.join("TCR_Alleles_tags_file.csv")
# label_df = pd.read_csv(train_tag_file_path, index_col=0)
#
# all_alleles = list(label_df.columns)
# print(set(all_alleles) - set(alleles_list))

def create_results_file():
    mypath = os.path.join("new_hla_results")
    # mypath = "."
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # onlyfiles = ["allele_B53_concat_graph_and_values_val_mean_0.475_test_mean_0.866.csv"]
    results_dic = {}
    for file in onlyfiles:
        allele = file.split("_")[1]
        print(allele)
        results_df = pd.read_csv(os.path.join(mypath, file), index_col=0)
        results_dic[allele] = float(results_df.loc["All auc"].loc["Run0"])
        print(results_dic[allele])
        x=1
    results_dic_df = pd.DataFrame.from_dict(results_dic, orient="index")
    return results_dic_df

results_dic_df = create_results_file()
print(results_dic_df)
results_dic_df.to_csv("new_Hla_TCR.csv")
