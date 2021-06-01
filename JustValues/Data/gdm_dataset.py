from torch.utils.data import Dataset
import pandas as pd
from torch import Tensor


class GDMDataset(Dataset):
    def __init__(self, data_file_path, tag_file_path):
        self._df = pd.read_csv(data_file_path, index_col='ID')
        self._tags = pd.read_csv(tag_file_path, index_col='ID')
        self._tags['Tag'] = self._tags['Tag'].astype(int)
        self._tags['trimester'] = self._tags['trimester'].astype(int)

        self._df = self._df[self._tags['trimester'] > 2]
        self._tags = self._tags[self._tags['trimester'] > 2]
        self._df.sort_index(inplace=True)
        self._tags.sort_index(inplace=True)
        del self._tags['trimester']

    def __getitem__(self, index):
        item = list(self._df.iloc[index])
        label = int(self._tags.iloc[index])
        return Tensor(item), label

    def __len__(self):
        a, b = self._df.shape
        return a

    def get_vector_size(self):
        a, b = self._df.shape
        return b

    def count_each_class(self):
        counter_dict = self._tags.value_counts()
        return counter_dict[0], counter_dict[1]
