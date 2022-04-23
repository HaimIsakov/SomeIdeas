import os
import pandas as pd

origin_dir = os.path.join("Data", "split_datasets_new")


class MyDatasets:
    def __init__(self, datasets_dict):
        self.datasets_dict = datasets_dict

    def get_dataset_files_no_external_test(self, dataset_name):
        # if dataset_name not in self.datasets_dict:
        #     print("Dataset is not in global datasets dictionary")
        # return self.datasets_dict[dataset_name]()
        # For no external_test
        subject_list = []
        if dataset_name == "abide":
            train_val_test_data_file_path = os.path.join("Data", "rois_ho")
            train_val_test_label_file_path = "Phenotypic_V1_0b_preprocessed1.csv"
            phenotype_df = pd.read_csv(train_val_test_label_file_path)
            subject_list = [value for value in phenotype_df["FILE_ID"].tolist() if value != "no_filename"]
        elif dataset_name == "cancer":
            train_val_test_data_file_path = os.path.join("cancer_data", "new_cancer_data.csv")  # It contains both train and test set
            train_val_test_label_file_path = os.path.join("cancer_data", "new_cancer_label.csv")  # It contains both train and test set
            adj_mat_path = "new_cancer_adj_matrix.csv"
            subject_list = range(11070)
        elif dataset_name == "ISB":
            train_val_test_data_file_path = os.path.join("covid", "new_ISB")
            train_val_test_label_file_path = os.path.join("covid", "ISB_samples.csv")
            label_df = pd.read_csv(train_val_test_label_file_path)
            label_df["sample"] = label_df["sample"] + "_" + label_df['status']
            label_df.set_index("sample", inplace=True)
            subject_list = list(label_df.index)
        elif dataset_name == "NIH":
            train_val_test_data_file_path = os.path.join("covid", "new_NIH")
            train_val_test_label_file_path = os.path.join("covid", "NIH_samples.csv")
            label_df = pd.read_csv(train_val_test_label_file_path)
            label_df["sample"] = label_df["sample"] + "_" + label_df['status']
            label_df.set_index("sample", inplace=True)
            subject_list = list(label_df.index)
        else:
            train_val_test_data_file_path = os.path.join("split_datasets_new", f"{dataset_name}_split_dataset",
                                                    f"OTU_merged_{dataset_name}"
                                                    f"_after_mipmlp_taxonomy_7_group_sub PCA_epsilon_1_normalization_log_"
                                                    f"After_mean_zeroing_after_arrangement.csv")
            train_val_test_label_file_path = os.path.join("split_datasets_new", f"{dataset_name}_split_dataset",
                                                          f"tag_{dataset_name}_file.csv")
        return train_val_test_data_file_path, train_val_test_label_file_path, subject_list

    def get_dataset_files_yes_external_test(self, dataset_name):
        subject_list = []
        train_subject_list, test_subject_list = [], []
        if dataset_name == "abide":
            train_data_file_path = os.path.join("Data", "rois_ho", "final_sample_files", "Final_Train")
            train_tag_file_path = os.path.join("Data", "Phenotypic_V1_0b_preprocessed1.csv")
            test_data_file_path = os.path.join("Data", "rois_ho", "final_sample_files", "Final_Test")
            test_tag_file_path = os.path.join("Data", "Phenotypic_V1_0b_preprocessed1.csv")
            for subdir, dirs, files in os.walk(train_data_file_path):
                for file in files:
                    file_id = file.split("_rois_ho")[0]
                    train_subject_list.append(file_id)
            for subdir, dirs, files in os.walk(test_data_file_path):
                for file in files:
                    file_id = file.split("_rois_ho")[0]
                    test_subject_list.append(file_id)
        elif dataset_name == "cancer":
            train_val_test_data_file_path = os.path.join("cancer_data", "new_cancer_data.csv")  # It contains both train and test set
            train_val_test_label_file_path = os.path.join("cancer_data", "new_cancer_label.csv")  # It contains both train and test set
            adj_mat_path = "new_cancer_adj_matrix.csv"
            subject_list = range(11070)
            raise NotImplementedError
        elif dataset_name == "tcr":
            # train_data_file_path = os.path.join("TCR_dataset", "final_sample_files", "Final_Train")
            train_data_file_path = os.path.join("TCR_Dataset2", "Train")
            # train_data_file_path = os.path.join("TCR_Dataset2", "Transpose_Train")

            train_tag_file_path = os.path.join("TCR_dataset", "samples.csv")
            # test_data_file_path = os.path.join("TCR_dataset", "final_sample_files", "Final_Test")
            test_data_file_path = os.path.join("TCR_Dataset2", "Test")
            # test_data_file_path = os.path.join("TCR_Dataset2", "Transpose_Test")

            test_tag_file_path = os.path.join("TCR_dataset", "samples.csv")
            # adj_mat_path = "distance_matrix.csv"
            label_df = pd.read_csv(train_tag_file_path)
            label_df["sample"] = label_df["sample"] + "_" + label_df['status']
            label_df.set_index("sample", inplace=True)
            train_subject_list = list(label_df[label_df["test/train"] == "train"].index)
            test_subject_list = list(label_df[label_df["test/train"] == "test"].index)
        elif dataset_name == "gdm":
            train_data_file_path = os.path.join("ShaharGdm", "train_val_set_gdm.csv")
            train_tag_file_path = os.path.join("ShaharGdm", "train_val_set_gdm_tags.csv")
            test_data_file_path = os.path.join("ShaharGdm", "test_set_gdm.csv")
            test_tag_file_path = os.path.join("ShaharGdm", "test_set_gdm_tags.csv")
            train_label_df = pd.read_csv(train_tag_file_path, index_col=0)
            test_label_df = pd.read_csv(test_tag_file_path, index_col=0)
            train_subject_list = list(train_label_df.index)
            test_subject_list = list(test_label_df.index)
        else:
            train_data_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
                                                f'train_val_set_{dataset_name}_microbiome.csv')
            train_tag_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
                                               f'train_val_set_{dataset_name}_tags.csv')

            test_data_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
                                               f'test_set_{dataset_name}_microbiome.csv')
            test_tag_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
                                              f'test_set_{dataset_name}_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path, \
               train_subject_list, test_subject_list

    def microbiome_files(self, dataset_name):
        print("origin_dir", origin_dir)
        # train_data_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset', f'train_val_set_{dataset_name}_microbiome_no_zero_cols.csv')
        train_data_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset', f'train_val_set_{dataset_name}_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset', f'train_val_set_{dataset_name}_tags.csv')

        # test_data_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset', f'test_set_{dataset_name}_microbiome_no_zero_cols.csv')
        test_data_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
                                           f'test_set_{dataset_name}_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset', f'test_set_{dataset_name}_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def gdm_files():
        train_data_file_path = os.path.join(origin_dir, 'GDM_split_dataset', 'train_val_set_gdm_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir, 'GDM_split_dataset', 'train_val_set_gdm_tags.csv')

        test_data_file_path = os.path.join(origin_dir, 'GDM_split_dataset', 'test_set_gdm_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir, 'GDM_split_dataset', 'test_set_gdm_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def cirrhosis_files():
        train_data_file_path = os.path.join(origin_dir,'Cirrhosis_split_dataset', 'train_val_set_Cirrhosis_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir,'Cirrhosis_split_dataset', 'train_val_set_Cirrhosis_tags.csv')

        test_data_file_path = os.path.join(origin_dir,'Cirrhosis_split_dataset', 'test_set_Cirrhosis_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir,'Cirrhosis_split_dataset', 'test_set_Cirrhosis_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def ibd_files():
        train_data_file_path = os.path.join(origin_dir,'IBD_split_dataset', 'train_val_set_IBD_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir,'IBD_split_dataset', 'train_val_set_IBD_tags.csv')

        test_data_file_path = os.path.join(origin_dir,'IBD_split_dataset', 'test_set_IBD_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir,'IBD_split_dataset', 'test_set_IBD_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def bw_files():
        train_data_file_path = os.path.join(origin_dir,'bw_split_dataset', 'train_val_set_bw_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir,'bw_split_dataset', 'train_val_set_bw_tags.csv')

        test_data_file_path = os.path.join(origin_dir,'bw_split_dataset', 'test_set_bw_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir,'bw_split_dataset', 'test_set_bw_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def ibd_chrone_files():
        train_data_file_path = os.path.join(origin_dir,'IBD_Chrone_split_dataset', 'train_val_set_IBD_Chrone_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir,'IBD_Chrone_split_dataset', 'train_val_set_IBD_Chrone_tags.csv')

        test_data_file_path = os.path.join(origin_dir,'IBD_Chrone_split_dataset', 'test_set_IBD_Chrone_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir,'IBD_Chrone_split_dataset', 'test_set_IBD_Chrone_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def allergy_or_not_files():
        train_data_file_path = os.path.join(origin_dir,'Allergy_or_not_split_dataset', 'train_val_set_Allergy_or_not_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir,'Allergy_or_not_split_dataset', 'train_val_set_Allergy_or_not_tags.csv')

        test_data_file_path = os.path.join(origin_dir,'Allergy_or_not_split_dataset', 'test_set_Allergy_or_not_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir,'Allergy_or_not_split_dataset', 'test_set_Allergy_or_not_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def allergy_milk_or_not_files():
        train_data_file_path = os.path.join(origin_dir,'Allergy_milk_split_dataset', 'train_val_set_Allergy_milk_or_not_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir,'Allergy_milk_split_dataset', 'train_val_set_Allergy_milk_or_not_tags.csv')

        test_data_file_path = os.path.join(origin_dir,'Allergy_milk_split_dataset', 'test_set_Allergy_milk_or_not_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir,'Allergy_milk_split_dataset', 'test_set_Allergy_milk_or_not_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def male_vs_female():
        train_data_file_path = os.path.join(origin_dir,'Male_vs_Female_split_dataset', 'train_val_set_Male_vs_Female_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir,'Male_vs_Female_split_dataset', 'train_val_set_Male_vs_Female_tags.csv')

        test_data_file_path = os.path.join(origin_dir,'Male_vs_Female_split_dataset', 'test_set_Male_vs_Female_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir,'Male_vs_Female_split_dataset', 'test_set_Male_vs_Female_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def male_vs_female_species():
        train_data_file_path = os.path.join(origin_dir,'Male_vs_Female_Species_split_dataset', 'train_val_set_Male_vs_Female_Species_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir,'Male_vs_Female_Species_split_dataset', 'train_val_set_Male_vs_Female_Species_tags.csv')

        test_data_file_path = os.path.join(origin_dir,'Male_vs_Female_Species_split_dataset', 'test_set_Male_vs_Female_Species_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir,'Male_vs_Female_Species_split_dataset', 'test_set_Male_vs_Female_Species_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def peanut():
        train_data_file_path = os.path.join(origin_dir, 'peanut_split_dataset', 'train_val_set_peanut_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir, 'peanut_split_dataset', 'train_val_set_peanut_tags.csv')

        test_data_file_path = os.path.join(origin_dir,'peanut_split_dataset', 'test_set_peanut_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir,'peanut_split_dataset', 'test_set_peanut_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def nut():
        train_data_file_path = os.path.join(origin_dir, 'nut_split_dataset', 'train_val_set_nut_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir, 'nut_split_dataset', 'train_val_set_nut_tags.csv')

        test_data_file_path = os.path.join(origin_dir,'nut_split_dataset', 'test_set_nut_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir,'nut_split_dataset', 'test_set_nut_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def nugent():
        train_data_file_path = os.path.join(origin_dir, 'nugent_split_dataset', 'train_val_set_nugent_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir, 'nugent_split_dataset', 'train_val_set_nugent_tags.csv')

        test_data_file_path = os.path.join(origin_dir, 'nugent_split_dataset', 'test_set_nugent_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir, 'nugent_split_dataset', 'test_set_nugent_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path


    @staticmethod
    def allergy_milk_no_controls():
        train_data_file_path = os.path.join(origin_dir, 'milk_no_controls_split_dataset', 'train_val_set_milk_no_controls_microbiome.csv')
        train_tag_file_path = os.path.join(origin_dir, 'milk_no_controls_split_dataset', 'train_val_set_milk_no_controls_tags.csv')

        test_data_file_path = os.path.join(origin_dir, 'milk_no_controls_split_dataset', 'test_set_milk_no_controls_microbiome.csv')
        test_tag_file_path = os.path.join(origin_dir, 'milk_no_controls_split_dataset', 'test_set_milk_no_controls_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path