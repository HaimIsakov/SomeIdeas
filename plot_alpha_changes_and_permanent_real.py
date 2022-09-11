import numpy as np
import pandas as pd
import seaborn as sns

# def bar_plot_with_std(gcn2_result_constant, gcn2_result_constant_std, gcn2_result_change, gcn2_result_change_std,
#                       gvm_result_constant, gvm_result_constant_std, gvm_result_change, gvm_result_change_std):




if __name__ == "__main__":
    gcn2_result_constant = [0.829, 0.981, 0.532, 0.892, 0.441, 0.971, 0.571, 0.47, 0.527]
    gcn2_result_constant_std = [0.043, 0.005, 0.04, 0.019, 0.053, 0.004, 0.064, 0.067, 0.07]

    gcn2_result_change = [0.867, 0.988, 0.631, 0.902, 0.44, 0.962, 0.718, 0.451, 0.509]
    gcn2_result_change_std = [0.01, 0.005, 0.029, 0.014, 0.056, 0.003, 0.067, 0.048, 0.041]

    gvm_result_constant = [0.842, 0.981, 0.541, 0.945, 0.436, 0.968, 0.6885, 0.463, 0.54]
    gvm_result_constant_std = [0.027, 0.009, 0.051, 0.013, 0.083, 0.003, 0.05, 0.05, 0.065]

    gvm_result_change = [0.834, 0.981, 0.555, 0.94, 0.523, 0.966, 0.765, 0.519, 0.541]
    gvm_result_change_std = [0.014, 0.01, 0.072, 0.013, 0.065, 0.005, 0.07, 0.043, 0.017]

    datasets = ["Cirrhosis", "IBD", "CA", "CD-IBD", "Male Female", "Nugent", "Milk", "Nut", "Peanut"]
    df = pd.DataFrame(data=np.transpose(np.array([gcn2_result_constant, gcn2_result_constant_std, gcn2_result_change, gcn2_result_change_std,
                      gvm_result_constant, gvm_result_constant_std, gvm_result_change, gvm_result_change_std])),
                      columns=["gcn2_result_constant", "gcn2_result_constant_std", "gcn2_result_change", "gcn2_result_change_std",
                      "gvm_result_constant", "gvm_result_constant_std", "gvm_result_change", "gvm_result_change_std"],
                      index=datasets)

    x=1