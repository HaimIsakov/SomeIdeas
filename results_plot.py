import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def heat_map(corr_mat):
    # np.fill_diagonal(corr_mat, 1)
    corr_mat.values[[np.arange(corr_mat.shape[0])] * 2] = 1

    ax = sns.heatmap(corr_mat, yticklabels=False, xticklabels=False)
    plt.tight_layout()
    plt.savefig("corr_mat_tcr.png")
    plt.show()

def histogram(corr_mat):
    corr_mat.stack().hist(grid=False, bins=300)
    plt.show()


def create_bar_plot(df, title):
    columns = list(df.columns)
    mean_df = df[[col for col in columns if "mean" in col]]
    std_df = df[[col for col in columns if "std" in col]]
    std_df.columns = mean_df.columns
    fig, ax = plt.subplots()
    b = mean_df.plot.bar(yerr=std_df,ax=ax,capsize=2)
    plt.xlabel("Dataset")
    plt.ylim((0,1.5))
    plt.ylabel("Auc/Alpha_value")
    save_file = f"summary_with_alpha_{title}"
    plt.title(save_file)
    plt.tight_layout(pad=1)
    plt.savefig(f'{save_file}.png')
    plt.show()

def create_bar_plot_new(df, rows2keep, cols2delete, title):
    df.drop(cols2delete, axis=1, inplace=True)
    df = df.loc[rows2keep]
    columns = list(df.columns)
    mean_cols = []
    std_cols = []
    for i, col in enumerate(columns):
        if i % 2 == 0:
            mean_cols.append(col)
        else:
            std_cols.append(col)
    mean_df = df[mean_cols]
    std_df = df[std_cols]
    std_df.columns = mean_df.columns
    fig, ax = plt.subplots()
    b = mean_df.plot.bar(yerr=std_df, ax=ax, capsize=0.5)
    plt.xlabel("Dataset")
    plt.ylim((0.4, 1))
    plt.ylabel("Auc")
    save_file = f"summary_{title}"
    plt.title(save_file)
    ax.legend(loc='center left', bbox_to_anchor=(0.05, 1.2), ncol=3, fancybox=True, shadow=True)

    plt.tight_layout(pad=1)
    plt.savefig(f'{save_file}.png')
    plt.show()


def concat_mean_std(df):
    for col in list(df.columns):
        if "std" not in col:
            print(col)
            df[col] = df.apply(lambda x: str(x[col]) + "±" + str(x[col +" (std)"]), axis=1)
    df.drop([col for col in df.columns if "std" in col], axis=1, inplace=True)
    df.to_csv("all_models_results_06_06_concat.csv")
    x=1

if __name__ == "__main__":
    # for mission in ["just_graph", "graph_and_values", "double_gcn_layer", "concat_graph_and_values"]:
    #     df = pd.read_csv(f"{mission}_all_datasets_results_train_val_test_09_04_2022.csv", index_col=0)
    #     create_bar_plot(df, f"{mission}")

    # corr_mat_df = pd.read_csv("tcr_corr_mat_125_with_sample_size_547_run_number_0.csv", index_col=0)
    # histogram(corr_mat_df)
    # heat_map(corr_mat_df)

    # df = pd.read_csv("all_models_06_06.csv", index_col=0)
    # rows2keep = ["IBD", "CD-IBD", "Nugent", "Cirrhosis", "BW", "Milk", "Nut", "Peanut", "MF", "abide",
    #              "TCRs"]
    # cols2delete = []
    # title = "all_models_06_06"
    # create_bar_plot_new(df, rows2keep, cols2delete, title)
    df = pd.read_csv("all_models_06_06.csv", index_col=0)
    concat_mean_std(df)
