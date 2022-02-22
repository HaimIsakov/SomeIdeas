import pandas as pd
from matplotlib import pyplot as plt


def plot_tcr_samples(df):
    x = df.plot()
    x.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
    x.xaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

    plt.xlabel("# Train samples")
    plt.ylabel("Auc on test set")
    plt.yticks([0.55 + 0.05 * i for i in range(9)])
    plt.savefig("Tcr dataset - Auc on test set vs number of samples in training set.pdf")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("tcr_samples_results.csv", index_col=0)
    plot_tcr_samples(df)
