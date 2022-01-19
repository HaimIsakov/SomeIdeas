import os
import pandas as pd
from sklearn.model_selection import train_test_split


def divide_to_train_test_dirs(directory, train_inx, test_inx):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            file_id = file.split("_rois_ho")[0]
            print(file_id)
            from_path = os.path.join(directory, file)
            if file_id in train_inx:
                to_path = os.path.join(directory, "final_sample_files", "Final_Train", file)
            elif file_id in test_inx:
                to_path = os.path.join(directory, "final_sample_files", "Final_Test", file)
            else:
                to_path = ""
                print("Error")
            os.replace(from_path, to_path)


if __name__ == "__main__":
    sample_file_path = "Phenotypic_V1_0b_preprocessed1.csv"
    sample_df = pd.read_csv(sample_file_path)
    subject_list = [value for value in sample_df["FILE_ID"].tolist() if value != "no_filename"]
    train_inx, test_inx = train_test_split(subject_list, test_size=0.2, random_state=42)
    directory = "rois_ho"
    divide_to_train_test_dirs(directory, train_inx, test_inx)
