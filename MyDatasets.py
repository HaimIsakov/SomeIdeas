import os


class MyDatasets:
    def __init__(self, datasets_dict):
        self.datasets_dict = datasets_dict

    def get_dataset_files(self, dataset_name):
        if dataset_name not in self.datasets_dict:
            print("Dataset is not in global datasets dictionary")
        return self.datasets_dict[dataset_name]()

    @staticmethod
    def gdm_files():
        train_data_file_path = os.path.join('GDM_split_dataset', 'train_val_set_gdm_microbiome.csv')
        train_tag_file_path = os.path.join('GDM_split_dataset', 'train_val_set_gdm_tags.csv')

        test_data_file_path = os.path.join('GDM_split_dataset', 'test_set_gdm_microbiome.csv')
        test_tag_file_path = os.path.join('GDM_split_dataset', 'test_set_gdm_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def cirrhosis_files():
        train_data_file_path = os.path.join('Cirrhosis_split_dataset', 'train_val_set_Cirrhosis_microbiome.csv')
        train_tag_file_path = os.path.join('Cirrhosis_split_dataset', 'train_val_set_Cirrhosis_tags.csv')

        test_data_file_path = os.path.join('Cirrhosis_split_dataset', 'test_set_Cirrhosis_microbiome.csv')
        test_tag_file_path = os.path.join('Cirrhosis_split_dataset', 'test_set_Cirrhosis_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def ibd_files():
        train_data_file_path = os.path.join('IBD_split_dataset', 'train_val_set_IBD_microbiome.csv')
        train_tag_file_path = os.path.join('IBD_split_dataset', 'train_val_set_IBD_tags.csv')

        test_data_file_path = os.path.join('IBD_split_dataset', 'test_set_IBD_microbiome.csv')
        test_tag_file_path = os.path.join('IBD_split_dataset', 'test_set_IBD_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def bw_files():
        train_data_file_path = os.path.join('Black_vs_White_split_dataset', 'train_val_set_Black_vs_White_microbiome.csv')
        train_tag_file_path = os.path.join('Black_vs_White_split_dataset', 'train_val_set_Black_vs_White_tags.csv')

        test_data_file_path = os.path.join('Black_vs_White_split_dataset', 'test_set_Black_vs_White_microbiome.csv')
        test_tag_file_path = os.path.join('Black_vs_White_split_dataset', 'test_set_Black_vs_White_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def ibd_chrone_files():
        train_data_file_path = os.path.join('IBD_Chrone_split_dataset', 'train_val_set_IBD_Chrone_microbiome.csv')
        train_tag_file_path = os.path.join('IBD_Chrone_split_dataset', 'train_val_set_IBD_Chrone_tags.csv')

        test_data_file_path = os.path.join('IBD_Chrone_split_dataset', 'test_set_IBD_Chrone_microbiome.csv')
        test_tag_file_path = os.path.join('IBD_Chrone_split_dataset', 'test_set_IBD_Chrone_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def allergy_or_not_files():
        train_data_file_path = os.path.join('Allergy_or_not_split_dataset', 'train_val_set_Allergy_or_not_microbiome.csv')
        train_tag_file_path = os.path.join('Allergy_or_not_split_dataset', 'train_val_set_Allergy_or_not_tags.csv')

        test_data_file_path = os.path.join('Allergy_or_not_split_dataset', 'test_set_Allergy_or_not_microbiome.csv')
        test_tag_file_path = os.path.join('Allergy_or_not_split_dataset', 'test_set_Allergy_or_not_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def allergy_milk_or_not_files():
        train_data_file_path = os.path.join('Allergy_milk_split_dataset', 'train_val_set_Allergy_milk_or_not_microbiome.csv')
        train_tag_file_path = os.path.join('Allergy_milk_split_dataset', 'train_val_set_Allergy_milk_or_not_tags.csv')

        test_data_file_path = os.path.join('Allergy_milk_split_dataset', 'test_set_Allergy_milk_or_not_microbiome.csv')
        test_tag_file_path = os.path.join('Allergy_milk_split_dataset', 'test_set_Allergy_milk_or_not_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

    @staticmethod
    def male_vs_female():
        train_data_file_path = os.path.join('Male_vs_Female_split_dataset', 'train_val_set_Male_vs_Female_microbiome.csv')
        train_tag_file_path = os.path.join('Male_vs_Female_split_dataset', 'train_val_set_Male_vs_Female_tags.csv')

        test_data_file_path = os.path.join('Male_vs_Female_split_dataset', 'test_set_Male_vs_Female_microbiome.csv')
        test_tag_file_path = os.path.join('Male_vs_Female_split_dataset', 'test_set_Male_vs_Female_tags.csv')
        return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path
