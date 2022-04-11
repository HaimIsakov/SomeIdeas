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


if __name__ == "__main__":
    # for mission in ["just_graph", "graph_and_values", "double_gcn_layer", "concat_graph_and_values"]:
    #     df = pd.read_csv(f"{mission}_all_datasets_results_train_val_test_09_04_2022.csv", index_col=0)
    #     create_bar_plot(df, f"{mission}")

    corr_mat_df = pd.read_csv("corr_mat_tcr_new.csv", index_col=0)

    heat_map(corr_mat_df)
