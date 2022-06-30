import os
import sys
for path_name in [os.path.join(os.path.dirname(__file__)),
                  os.path.join(os.path.dirname(__file__), 'Data'),
                  os.path.join(os.path.dirname(__file__), 'Missions')]:
    sys.path.append(path_name)

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


class Repertoires:
    def __init__(self, name, nump):
        self.name = name
        self.combinations = {}
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

    """
    creates the vector that tells us whether a person i is pos/neg
    """

    def create_cmv_vector(self):

        self.cmv = np.zeros(self.size, dtype=np.int8)
        for item, index in self.__id2patient.items():
            if item[1] == 'positive':
                self.cmv[index] = 1

    """
    Create the dictionary to which all the data is loaded to. For each TCR in the data, 
    A vector is created that contains for each repertoire the number of times the TCR appears 
    in the repertoire. The vector is saved in the combinations dictionary.
    """

    def create_combinations(self):
        print(f"\n{Fore.LIGHTBLUE_EX}Generate a quantity of {Fore.LIGHTMAGENTA_EX}instances combination{Fore.RESET}")
        start_time = time.time()
        for personal_info, ind in tqdm(self.__id2patient.items(), total=self.size, desc='Create Combinations',
                                       bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTBLUE_EX, Fore.RESET)):
            # reading each person's file to fill the appropriate place in the dict for that person
            if ind >= self.size:
                break
            _, _, path = personal_info
            df = pd.read_csv(path, usecols=['combined'])
            v_comb = df["combined"]
            for element in v_comb:
                if element not in self.combinations:
                    self.combinations[element] = np.zeros(self.size, dtype=np.int8)
                self.combinations[element][ind] += 1
        print(f"{Fore.LIGHTBLUE_EX}Amount of combinations: {Fore.RED}{len(self.combinations)}{Fore.RESET}")
        print(
            f"{Fore.LIGHTBLUE_EX}Generate a quantity of instances combinations, {Fore.RED}time elapsed: {time.time() - start_time:.2f}s{Fore.RESET}\n")

    def save_data(self, directory, files=None):
        self.personal_information(directory, files=files)
        self.create_combinations()
        self.create_cmv_vector()

    """
    calculated the chi squared score for each TCR in the data
    """

    def scatter_score(self):
        self.score = {}
        numn = np.count_nonzero(1 - self.cmv)
        nump = np.count_nonzero(self.cmv)
        pos_precent = nump / (numn + nump)
        print("Calculating chi squared score")
        for element, val in tqdm(self.combinations.items()):
            sumrec = np.count_nonzero(val)
            if sumrec < 2:
                self.score[element] = 0
            else:
                sumPos = np.dot(np.sign(val), self.cmv)
                self.score[element] = abs(sumPos - pos_precent * sumrec) * (sumPos - pos_precent * sumrec) / (
                        pos_precent * sumrec)
            if abs(self.score[element]) > 50:
                del self.score[element]

    '''
    Finds the numrec most reactive TCRs, with the largest chi squared score. 
    The actual number or reactive TCRs might be slightly larger that numrec if there are TCRs with exactly the same chi squared score. 
    '''

    def new_outlier_finder(self, numrec, pickle_name="outliers"):
        self.scatter_score()
        for element, score in self.score.items():
            self.score[element] = abs(score)
        self.score = dict(sorted(self.score.items(), key=lambda item: item[1], reverse=True))
        element, min_score = list(self.score.items())[numrec]
        print("Reactive TCRs found:")
        for element, score in list(self.score.items()):
            if score >= min_score:
                print(f'element: {element}, score: {score}')
                self.outlier[element] = self.combinations[element]
        with open(f"{pickle_name}.pkl", "wb") as f:
            pickle.dump(self.outlier, f)
        print("Number of outliers", len(self.outlier))
        return self.outlier

