import numpy as np
import pandas as pd


def run_again_from_nni_results_csv(file, n_rows=10):
    result_df = pd.read_csv(file, header=0)
    result_df.sort_values(by=['reward'], inplace=True, ascending=False)
    del result_df["trialJobId"]
    del result_df["intermediate"]
    del result_df["reward"]
    first_n_rows = result_df[0:n_rows]
    params_list = [{} for i in range(n_rows)]
    for i in range(n_rows):
        for j in first_n_rows.columns:
            params_list[i][j] = int(first_n_rows.iloc[i][j]) if type(first_n_rows.iloc[i][j]) is \
                                                                np.int64 else first_n_rows.iloc[i][j]
    return params_list


def run_again_from_nni_results_csv_format2(file, n_rows=10):
    result_df = pd.read_csv(file, header=0, index_col=0)
    result_df.sort_values(by=['mean_val_nni'], inplace=True, ascending=False)
    del result_df["std_val_nni"]
    del result_df["mean_test_nni"]
    del result_df["std_test_nni"]
    del result_df["mean_val_nni"]
    first_n_rows = result_df[0:n_rows]
    params_list = [{} for i in range(n_rows)]
    for i in range(n_rows):
        for j in first_n_rows.columns:
            params_list[i][j] = int(first_n_rows.iloc[i][j]) if type(first_n_rows.iloc[i][j]) is \
                                                                np.int64 else first_n_rows.iloc[i][j]
    return params_list


