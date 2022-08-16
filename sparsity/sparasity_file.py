import os
import MIPMLP
import pandas as pd


def check_bacterias_two_dataframes(df1, df2):
    sub = []
    mean = []
    for i in df1.columns:
        if i[-2:] in ['_0','_1','_2']:
            i = i[:-2]
        sub.append(i.replace('_0;', ";").replace('_1;', ";").replace('_2;', ";"))
    mean = list(df2.columns)
    # for i in df2.columns:
    #     i = i.split(';')
    #     while len(i) > 0 and i[-1][-1] == "_":
    #         i = i[:-1]
    #     i = ';'.join(i)
    #     if i != '':
    #         mean.append(i)

    sub = set(sub)
    mean = set(mean)
    print(len(sub))
    print(len(mean))
    print(sub == mean)
    print("Is mean is subset of sub", mean.issubset(sub))
    print(sorted(mean - sub))
    print(sorted(sub - mean))
    # print(sorted(sub))
    # print(sorted(mean))


def create_microbiome_file(subpca_df, mean_df, result_dir):
    # subpca_df = pd.read_csv(subpca_file, index_col=0)
    # mean_df = pd.read_csv(mean_file, index_col=0)
    subpca_df.sort_index(axis=1, inplace=True)
    mean_df.sort_index(axis=1, inplace=True)
    subpca_df.index = subpca_df.index.map(str)
    mean_df.index = mean_df.index.map(str)
    subpca_df.sort_index(axis=0, inplace=True)
    mean_df.sort_index(axis=0, inplace=True)
    # check_bacterias_two_dataframes(subpca_df, mean_df)
    sub = list(subpca_df.columns)
    mean = list(mean_df.columns)
    for sub_col, mean_col in zip(sub, mean):
        if sub_col == "ID" or mean_col == "ID":
            continue
        subpca_df[sub_col] = subpca_df[sub_col].where(mean_df[mean_col] != 0, 0.0)

    # subpca_df.to_csv(os.path.join(result_dir, subpca_file.split("/")[1].replace(".csv", "") + "_After_mean_zeroing" + ".csv"))

preprocess_prms1 = {'taxonomy_level': 7, 'taxnomy_group': 'sub PCA', 'epsilon': 1,
                    'normalization': 'log', 'z_scoring': 'No', 'norm_after_rel': 'No',
                    'std_to_delete': 0, 'pca': (0, 'PCA'), "rare_bacteria_threshold": -1}
# Relative normalization
preprocess_prms2 = {'taxonomy_level': 7, 'taxnomy_group': 'mean', 'epsilon': 1,
                    'normalization': 'relative', 'z_scoring': 'No', 'norm_after_rel': 'No',
                    'std_to_delete': 0, 'pca': (0, 'PCA'), "rare_bacteria_threshold": -1}

df1 = pd.read_csv(os.path.join("Cirrhosis", "4sparse_75.csv"))
df2 = pd.read_csv(os.path.join("Cirrhosis", "4sparse_75.csv"))
mean_df = MIPMLP.preprocess(df1, tag=None, taxonomy_level=7,taxnomy_group='mean',epsilon=1,normalization='log',
                  z_scoring='No',norm_after_rel='No',pca= (0, 'PCA'))
sub_pca_df = MIPMLP.preprocess(df2, tag=None, taxonomy_level=7,taxnomy_group='sub PCA',epsilon=1,normalization='log',
                  z_scoring='No',norm_after_rel='No',pca= (0, 'PCA'))

mean_df.to_csv("mean_df.csv")

sub_pca_df.to_csv("sub_pca_df.csv")

x=1
create_microbiome_file(sub_pca_df, mean_df, "")

