import pandas as pd
import os


def iterate_files(directory, tcr_list, result_dir):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            print(file)
            path = os.path.join(directory, file)
            sample_df = pd.read_csv(path)
            intersec = set(tcr_list) & set(sample_df["combined"])
            new_sample_df = sample_df[["combined", "frequency"]]
            new_sample_df.set_index("combined", inplace=True)
            new_sample_df_only_tcr_list = new_sample_df.loc[intersec]
            new_sample_df_only_tcr_list.to_csv(os.path.join(result_dir, f"final_{file}"))


def get_tcr_list(adj_mat_path):
    adj_mat = pd.read_csv(adj_mat_path, index_col=0)
    print(adj_mat)
    return list(adj_mat.index)


if __name__ == "__main__":
    adj_mat_path = "distance_matrix.csv"
    directory = os.path.join("Train")
    result_dir = os.path.join("final_sample_files", "Final_Train")
    tcr_list = get_tcr_list(adj_mat_path)
    iterate_files(directory, tcr_list, result_dir)
