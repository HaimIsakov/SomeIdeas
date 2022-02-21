import os
import pickle
import time
import numpy as np
import pandas as pd
from colorama import Fore
from pathlib import Path
import random
from tqdm import tqdm

DIR = 'Graphs'
ADDED = "1_layer"
THRESHOLDS = {}
DATA = {}


class HistoMaker:
    def __init__(self, name, nump):
        self.name = name
        self.combinations = {}
        self.clustercom = {}
        self.__id2patient = {}
        self.score = {}
        self.outlier = {}
        self.size = nump
        self.cmv = None

    def personal_information(self, directory, files=None, mix=False):
        """
            creates the dict the connects people to their index
        """
        directory = Path(directory)
        # self.__size = len(list(directory.glob('*')))
        if files is None:
            files = list(directory.glob('*'))
        else:
            self.size = min(self.size, len(files))
        if self.name == "train":
            samples = random.sample(range(len(files)), self.size)
        else:
            samples = range(77)
        count = 0
        print("len of files", len(files))
        for ind, item in tqdm(enumerate(files), total=self.size, desc="Maintain patient order",
                              bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):

            if ind not in samples:
                continue

            # attr hold the patient and the path to him
            attr = item.stem.split('_')
            attr.append(item.as_posix())
            self.__id2patient[tuple(attr)] = count
            count += 1

    def create_cmv_vector(self):
        """
        creates the vector that tells us whether person i is pos/neg
        """
        self.cmv = np.zeros(self.size, dtype=np.int8)
        for item, index in self.__id2patient.items():
            # print(item[0], self.name, item[1])
            DATA[item[0]] = item[1]
            if item[1] == 'positive':
                self.cmv[index] = 1

    def create_combinations(self):
        """
        The dictionary to which data is loaded from all files contains all combinations,
        reads files from another dictionary.
        Initializes each element in the dictionary by key (combination) and the value by  zeros array
        in size of the patient.
        A key word already appears in the dictionary, so one patient's index is gathered.
        """
        print(f"\n{Fore.LIGHTBLUE_EX}Generate a quantity of {Fore.LIGHTMAGENTA_EX}instances combination{Fore.RESET}")
        start_time = time.time()
        for personal_info, ind in tqdm(self.__id2patient.items(), total=self.size, desc='Create Combinations',
                                       bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            # reading each person's file to fill the appropriate place in the dict for that person
            if ind >= self.size:
                break
            print('personal info', personal_info)
            _, _, path = personal_info
            df = pd.read_csv(path, usecols=['frequency', 'combined', 'autoencoder_projection'])
            v_frac, v_comb, v_vec = df["frequency"], df["combined"], df["autoencoder_projection"]
            print('file', path)
            for freq, element, vector in zip(v_frac, v_comb, v_vec):
                # "cleaning" the projection vector
                vector = vector[2:-2]
                if '_' in vector:
                    vector = vector.replace("_", "")
                # creating and adding to the dict
                if (element, vector) not in self.combinations:
                    self.combinations[(element, vector)] = (
                        np.zeros(self.size, dtype=np.int8), None)  # np.zeros(self.size, dtype=np.float32))
                self.combinations[(element, vector)][0][ind] += 1
                # self.combinations[(element, vector)][1][ind] += freq
        print(f"{Fore.LIGHTBLUE_EX}Amount of combinations: {Fore.RED}{len(self.combinations)}{Fore.RESET}")
        print(
            f"{Fore.LIGHTBLUE_EX}Generate a quantity of instances combinations, {Fore.RED}time elapsed: {time.time() - start_time:.2f}s{Fore.RESET}\n")

    def save_data(self, directory, files=None):
        """this includes the whole process of storing the basic data
            """
        self.personal_information(directory, files=files)
        self.create_combinations()
        self.filter_dict()
        self.create_cmv_vector()
        '''
        if 'test' not in self.name:
            with open(f"{self.name}_{self.size}_mix.pkl", "wb") as f:
                pickle.dump(self.combinations, f)
        '''

    def filter_dict(self):
        """Initial filtering of combinations dictionary, filtering is done by the number of
                impressions of a key if it is less than two The combination comes out of a dictionary.
                """
        print(
            f"{Fore.LIGHTBLUE_EX}Filter dictionary by combining multiple instances, {Fore.LIGHTMAGENTA_EX}at least seven{Fore.RESET}")
        start_time = time.time()
        print(f"length of combos{len(self.combinations)}")

        # if 'test' not in self.name:
        #     for k in tqdm(list(self.combinations.keys()), total=len(self.combinations), desc='Filter dictionary',
        #                   bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
        #         if np.count_nonzero(self.combinations[k][0]) < 7:
        #             del self.combinations[k]

        print(
            f"{Fore.LIGHTBLUE_EX}The amount of combination after the filtering operation: {Fore.RED}{len(self.combinations)}{Fore.RESET}")

        print(
            f"{Fore.LIGHTBLUE_EX}Filter dictionary by combining multiple instances at least seven, {Fore.RED}time elapsed: {time.time() - start_time:.2f}s{Fore.RESET}\n")

    def scatter_score(self):
        """
        creates a score of how much more prevelant positive people are for a certain recpetor
        """
        self.score = {}
        numn = np.count_nonzero(1 - self.cmv)
        nump = np.count_nonzero(self.cmv)
        pos_precent = nump / (numn + nump)
        for element, val in self.combinations.items():
            sumrec = np.count_nonzero(val[0])
            if sumrec < 2:
                self.score[element] = 0
            else:
                sumPos = np.dot(np.sign(val[0]), self.cmv)
                self.score[element] = abs(sumPos - pos_precent * sumrec) * (sumPos - pos_precent * sumrec) / (
                        pos_precent * sumrec)
            if abs(self.score[element]) > 50:
                del self.score[element]

    # def outlier_finder(self, run_number, cutoff=8.0, numrec=None):
    #     # if len(self.outlier) != 0:
    #     #    return
    #     # if os.path.isfile(f"outliers_{run_number}.pkl"):
    #     #     print(f"outliers_{run_number}.pkl is found")
    #     #     return
    #     self.scatter_score()
    #     for element, score in self.score.items():
    #         if abs(score) > cutoff:
    #             self.outlier[element] = self.combinations[element]
    #     if numrec is not None and len(self.outlier) < numrec:
    #         self.outlier_finder(run_number, cutoff=cutoff - 0.1, numrec=numrec)
    #     with open(f"outliers_{run_number}.pkl", "wb") as f:
    #         pickle.dump(self.outlier, f)
    #     # print(self.outlier)
    #     print(cutoff)
    #     print("Number of outliers", len(self.outlier))
    #     return len(self.outlier)

    def new_outlier_finder(self, numrec, pickle_name="outliers"):
        self.scatter_score()
        for element, score in self.score.items():
            self.score[element] = abs(score)
        self.score = dict(sorted(self.score.items(), key=lambda item: item[1], reverse=True))
        element, min_score = list(self.score.items())[numrec]
        for element, score in list(self.score.items()):
            if score >= min_score:
                print(f'element: {element}, score: {score}')
                self.outlier[element] = self.combinations[element]
        with open(f"{pickle_name}.pkl", "wb") as f:
            pickle.dump(self.outlier, f)
        print("Number of outliers", len(self.outlier))
        return len(self.outlier)

    def get_cutoff(self, numrec, thresholds):
        for cutoff, num in thresholds.items():
            if num >= numrec:
                return cutoff


# if __name__ == "__main__":
#     train = HistoMaker("train", 684)
#     file_directory_path = os.path.join("TCR_Dataset2", "Train")
#     files = list(Path(file_directory_path).glob('*'))
#     print(files)
#     train.save_data(file_directory_path, files=files)
#     numrec = 125
#     train.outlier_finder(numrec=numrec)
    # with open('thresholds100000.pkl', 'rb') as handle:
    #     threshold_dict = pickle.load(handle)
    # cutoff = train.get_cutoff(numrec, threshold_dict)
    # train.outlier_finder(cutoff=cutoff, numrec=numrec)
