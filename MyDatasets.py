import os

import pandas as pd

origin_dir = "split_datasets_new"


class MyDatasets:
    def __init__(self, datasets_dict):
        self.datasets_dict = datasets_dict

    def get_dataset_files(self, dataset_name):
        # if dataset_name not in self.datasets_dict:
        #     print("Dataset is not in global datasets dictionary")
        # return self.datasets_dict[dataset_name]()
        subject_list = []
        if dataset_name == "abide":
            train_val_test_data_file_path = "rois_ho"
            train_val_test_label_file_path = "Phenotypic_V1_0b_preprocessed1.csv"
            phenotype_df = pd.read_csv(train_val_test_label_file_path)
            subject_list = [value for value in phenotype_df["FILE_ID"].tolist() if value != "no_filename"]
        elif dataset_name == "cancer":
            train_val_test_data_file_path = os.path.join("cancer_data", "new_cancer_data.csv")  # It contains both train and test set
            train_val_test_label_file_path = os.path.join("cancer_data", "new_cancer_label.csv")  # It contains both train and test set
            adj_mat_path = "new_cancer_adj_matrix.csv"
            subject_list = range(11070)
        else:
            train_val_test_data_file_path = os.path.join("split_datasets_new", f"{dataset_name}_split_dataset",
                                                    f"OTU_merged_{dataset_name}"
                                                    f"_after_mipmlp_taxonomy_7_group_sub PCA_epsilon_1_normalization_log_"
                                                    f"After_mean_zeroing_after_arrangement.csv")
            train_val_test_label_file_path = os.path.join("split_datasets_new", f"{dataset_name}_split_dataset",
                                                          f"tag_{dataset_name}_file.csv")
        return train_val_test_data_file_path, train_val_test_label_file_path, subject_list

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
