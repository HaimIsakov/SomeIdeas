import numpy as np
import pandas as pd
import seaborn as sns

# def bar_plot_with_std(gcn2_result_constant, gcn2_result_constant_std, gcn2_result_change, gcn2_result_change_std,
#                       gvm_result_constant, gvm_result_constant_std, gvm_result_change, gvm_result_change_std):
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

if __name__ == "__main__":
    gcn2_result_constant = np.array([0.829, 0.981, 0.532, 0.892, 0.441, 0.971, 0.571, 0.47, 0.527])
    gcn2_result_constant_std = np.array([0.043, 0.005, 0.04, 0.019, 0.053, 0.004, 0.064, 0.067, 0.07])/(10**0.5)

    gcn2_result_change = np.array([0.867, 0.988, 0.631, 0.902, 0.44, 0.962, 0.718, 0.451, 0.509])
    gcn2_result_change_std = np.array([0.01, 0.005, 0.029, 0.014, 0.056, 0.003, 0.067, 0.048, 0.041])/(10**0.5)

    gvm_result_constant = np.array([0.842, 0.981, 0.541, 0.945, 0.436, 0.968, 0.751, 0.463, 0.54])
    gvm_result_constant_std = np.array([0.027, 0.009, 0.051, 0.013, 0.083, 0.003, 0.05, 0.05, 0.065])/(10**0.5)

    gvm_result_change = np.array([0.834, 0.981, 0.555, 0.94, 0.523, 0.966, 0.765, 0.519, 0.541])
    gvm_result_change_std = np.array([0.014, 0.01, 0.072, 0.013, 0.065, 0.005, 0.07, 0.043, 0.017])/(10**0.5)

    datasets = ["Cirrhosis", "IBD", "CA", "CD-IBD", "Male Female", "Nugent", "Milk", "Nut", "Peanut"]
    # df = pd.DataFrame(data=np.transpose(np.array([gcn2_result_constant, gcn2_result_constant_std, gcn2_result_change, gcn2_result_change_std,
    #                   gvm_result_constant, gvm_result_constant_std, gvm_result_change, gvm_result_change_std])),
    #                   columns=["gcn2_result_constant", "gcn2_result_constant_std", "gcn2_result_change", "gcn2_result_change_std",
    #                   "gvm_result_constant", "gvm_result_constant_std", "gvm_result_change", "gvm_result_change_std"],
    #                   index=datasets)
    # plt.rcParams["figure.figsize"] = (13,10)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
    # fig, ax = plt.subplots(nrows=1, ncols=1)

    df1 = pd.DataFrame(data=np.transpose(np.array([gcn2_result_constant, gcn2_result_change])),
                      columns=["Constant α", "Learnt α"],
                      index=datasets)
    df2 = pd.DataFrame(data=np.transpose(np.array([gvm_result_constant, gvm_result_change])),
                       columns=["Constant α", "Learnt α"],
                       index=datasets)

    # df1.plot(kind='bar', yerr=[gcn2_result_constant_std, gcn2_result_change_std], ax=ax1, title="GCN2")
    # df2.plot(kind='bar', yerr=[gvm_result_constant_std, gvm_result_change_std], ax=ax2, title="GVM")
    labels = list(df1.index)
    width = 0.35
    ax1.bar(labels, gcn2_result_constant, width, yerr=gcn2_result_constant_std, label='Constant α')
    ax1.bar(labels, gcn2_result_change - gcn2_result_constant, width, yerr=gcn2_result_change_std, bottom=gcn2_result_constant,
           label='Learnt α')

    ax2.bar(labels, gvm_result_constant, width, yerr=gvm_result_constant_std, label='Constant α')
    ax2.bar(labels, gvm_result_change - gvm_result_constant, width, yerr=gvm_result_change_std, bottom=gvm_result_constant,
        label='Learnt α')
    # plt.setp([ax1, ax2], ylabel='AUC')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

    ax1.set_ylabel("AUC")
    ax1.set_title("GCN2")
    ax2.set_title("GVM")

    # plt.xticks(rotation=45)
    ax1.legend(loc="best")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig("comapre_alpha_changes_and_constant_real_datasets")
    plt.show()
    x=1
