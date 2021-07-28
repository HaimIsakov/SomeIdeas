import pandas as pd


class MicrobiomeDataset:
    def __init__(self, data_file_path, tag_file_path):
        self.microbiome_df = pd.read_csv(data_file_path, index_col='ID')
        self.tags = pd.read_csv(tag_file_path, index_col='ID')  # has the tag and the group
        self.graphs_list = []
        self.groups, self.labels = [], []  # for "sklearn.model_selection.GroupShuffleSplit, Stratify"
        self.arrange_dataframes()
        # self.node_order = self.set_node_order()

    def arrange_dataframes(self):
        # self.remove_na()
        self.tags['Tag'] = self.tags['Tag'].astype(int)
        self.microbiome_df.sort_index(inplace=True)
        self.tags.sort_index(inplace=True)
        self.add_groups()  # It is important to verify that the order of instances is correct
        self.add_labels()

    def __len__(self):
        a, b = self.microbiome_df.shape
        return a

    def add_groups(self):
        self.groups = list(self.tags['Group'])

    def get_groups(self, indexes):
        return [self.groups[i] for i in indexes]

    def add_labels(self):
        self.labels = list(self.tags['Tag'])

    def get_label(self, i):
        return self.labels[i]

    def get_leaves_values(self, i):
        return list(self.microbiome_df.iloc[i])

    # def remove_na(self):
    #     index = self.tags['Tag'].index[self.tags['Tag'].apply(np.isnan)]
    #     self.tags.drop(index, inplace=True)
    #     self.microbiome_df.drop(index, inplace=True)

    def count_each_class(self):
        counter_dict = self.tags['Tag'].value_counts()
        return counter_dict[0], counter_dict[1]

    def get_leaves_number(self):
        a, b = self.microbiome_df.shape
        return b
