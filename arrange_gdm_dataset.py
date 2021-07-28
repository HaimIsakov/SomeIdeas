import pandas as pd
import numpy as np


class ArrangeGDMDataset:
    def __init__(self, data_file_path, tag_file_path, mission):
        self.mission = mission
        self._microbiome_df = pd.read_csv(data_file_path, index_col='ID')
        self._tags = pd.read_csv(tag_file_path, index_col='ID')
        self.graphs_list = []
        self.groups, self.labels = [], []  # for "sklearn.model_selection.GroupShuffleSplit, Stratify"
        lambda_func_trimester = lambda x: True  # lambda_func_trimester = lambda x: x == 1
        lambda_func_repetition = lambda x: True  # lambda_func_repetition = lambda x: x == 1
        self.arrange_dataframes(lambda_func_repetition=lambda_func_repetition, lambda_func_trimester=lambda_func_trimester)
        # self.node_order = self.set_node_order()

    def arrange_dataframes(self, lambda_func_repetition=lambda x: True, lambda_func_trimester=lambda x: True):
        self.remove_na()
        self._tags['Tag'] = self._tags['Tag'].astype(int)
        self.split_id_col()
        self._tags['trimester'] = self._tags['trimester'].astype(int)
        self._microbiome_df = self._microbiome_df[self._tags['trimester'].apply(lambda_func_trimester)]
        self._tags = self._tags[self._tags['trimester'].apply(lambda_func_trimester)]
        self._tags = self._tags[self._microbiome_df['Repetition'].apply(lambda_func_repetition)]
        self._microbiome_df = self._microbiome_df[self._microbiome_df['Repetition'].apply(lambda_func_repetition)]
        self._microbiome_df.sort_index(inplace=True)
        self._tags.sort_index(inplace=True)
        self.add_groups()  # It is important to verify that the order of instances is correct
        self.add_labels()
        del self._tags['trimester']
        del self._microbiome_df['Repetition']
        del self._microbiome_df['Code']

    def __len__(self):
        a, b = self._microbiome_df.shape
        return a

    def split_id_col(self):
        self._microbiome_df['Code'] = [cur_id.split('-')[0] for cur_id in self._microbiome_df.index]
        self._microbiome_df['Repetition'] = [int(cur_id.split('-')[-1]) for cur_id in self._microbiome_df.index]
        self._tags['Code'] = [cur_id.split('-')[0] for cur_id in self._tags.index]
        self._tags['trimester'] = [cur_id.split('-')[2][-1] for cur_id in self._tags.index]

    def add_groups(self):
        self.groups = list(self._microbiome_df['Code'])

    def get_groups(self, indexes):
        return [self.groups[i] for i in indexes]

    def add_labels(self):
        self.labels = list(self._tags['Tag'])

    def get_label(self, i):
        return self.labels[i]

    def get_leaves_values(self, i):
        return list(self._microbiome_df.iloc[i])

    def remove_na(self):
        index = self._tags['Tag'].index[self._tags['Tag'].apply(np.isnan)]
        self._tags.drop(index, inplace=True)
        self._microbiome_df.drop(index, inplace=True)

    def count_each_class(self):
        counter_dict = self._tags['Tag'].value_counts()
        return counter_dict[0], counter_dict[1]

    def get_leaves_number(self):
        a, b = self._microbiome_df.shape
        return b
