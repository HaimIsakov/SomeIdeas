from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from distance_matrix import *
from ofek_files_utils_functions import *
from train_test_val_one_time import TrainTestValOneTime
from train_test_val_ktimes_utils import *
from concat_graph_and_values import *
LOSS_PLOT = 'loss'
ACCURACY_PLOT = 'acc'
AUC_PLOT = 'auc'
TRAIN_JOB = 'train'
TEST_JOB = 'test'


class TrainTestValKTimes:
    def __init__(self, RECEIVED_PARAMS, device, train_val_dataset, test_dataset, **kwargs):
        # self.mission = mission
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.device = device
        self.train_val_dataset = train_val_dataset
        self.test_dataset = test_dataset
        self.kwargs = kwargs
        self.train_loader = None

    def train_group_k_cross_validation(self, k=5):
        dataset_len = len(self.train_val_dataset)
        train_metric, val_metric, test_metric, min_train_val_metric, alpha_list = [], [], [], [], []
        return_lists = [train_metric, val_metric, test_metric, min_train_val_metric, alpha_list]
        run = 0
        for i in range(k):
            indexes_array = np.array(range(dataset_len))
            train_idx, val_idx = train_test_split(indexes_array, test_size=0.2, shuffle=True)
            print(f"Run {run}")
            self.create_data_loaders(i, train_idx, val_idx)
            train_loader, val_loader, test_loader = self.create_data_loaders(i, train_idx, val_idx)

            model = get_model(self.train_val_dataset, self.RECEIVED_PARAMS, self.device)
            trainer_and_tester = TrainTestValOneTime(model, self.RECEIVED_PARAMS, train_loader, val_loader, test_loader,
                                                     self.device)

            early_stopping_results = start_training_process(trainer_and_tester, train_loader, val_loader, test_loader,
                                                            self.RECEIVED_PARAMS, self.device, self.train_val_dataset)
            min_val_train_auc = min(early_stopping_results['val_auc'], early_stopping_results['train_auc'])
            print("Minimum Validation and Train Auc", min_val_train_auc)
            min_train_val_metric.append(min_val_train_auc)  # the minimum between the aucs between train set and validation set
            train_metric.append(early_stopping_results['train_auc'])
            val_metric.append(early_stopping_results['val_auc'])
            test_metric.append(early_stopping_results['test_auc'])
            try:
                alpha_list.append(early_stopping_results['last_alpha_value'])
            except:
                pass
            run += 1
        return return_lists

    def compare_network(self, train_idx, i):
        print(self.kwargs)
        if "samples" not in self.kwargs:
            random_sample_from_train = len(train_idx)
        elif self.kwargs["samples"] == -1:
            random_sample_from_train = len(train_idx)
        else:
            random_sample_from_train = int(self.kwargs["samples"])
        print(f"\nTake only {random_sample_from_train} from the training set\n")
        mission = self.train_val_dataset.mission
        graph_type = self.kwargs["graph_model"]
        # print("Here, we ----do---- calculate again the golden-tcrs")
        train = Repertoires("train", random_sample_from_train)
        file_directory_path = os.path.join("..", "TCR_Dataset2", "Train")  # TCR_Dataset2 exists only in server
        # sample only some sample according to input sample size, and calc the golden tcrs only from them
        train_idx = random.sample(list(train_idx), random_sample_from_train)
        train_files = [Path(os.path.join(file_directory_path, self.train_val_dataset.subject_list[id] + ".csv"))
                       for id in train_idx]
        # print("Length of chosen files", len(train_files))
        numrec = int(self.RECEIVED_PARAMS["numrec"])  # cutoff is also a hyper-parameter
        # print("Number of golden-tcrs", numrec)
        train.save_data(file_directory_path, files=train_files)
        # save files' names
        outliers_pickle_name = f"new_graph_type_{graph_type}_tcr_outliers_{numrec}_with_sample_size_{len(train_files)}_run_number_{i}_mission_{mission}"
        adj_mat_path = f"new_graph_type_{graph_type}_tcr_mat_{numrec}_with_sample_size_{len(train_files)}_run_number_{i}_mission_{mission}"
        outlier = train.new_outlier_finder(numrec, pickle_name=outliers_pickle_name)  # find outliers and save to pickle
        corr_df_between_golden_tcrs = self.create_corr_tcr_network(train_idx, file_directory_path, outlier,
                                                                   f"new_graph_type_corr_tcr_corr_mat_{numrec}_with_sample_size_{len(train_files)}_run_number_{i}_mission_{mission}")
        proj_matrix = self.create_projection_tcr_network(outliers_pickle_name,
                                                         f"new_graph_type_proj_tcr_mat_{numrec}_with_sample_size_{len(train_files)}_run_number_{i}_mission_{mission}")

    def tcr_dataset_dealing(self, train_idx, i):
        print(self.kwargs)
        if "samples" not in self.kwargs:
            random_sample_from_train = len(train_idx)
        elif self.kwargs["samples"] == -1:
            random_sample_from_train = len(train_idx)
        else:
            random_sample_from_train = int(self.kwargs["samples"])
        print(f"\nTake only {random_sample_from_train} from the training set\n")
        mission = self.train_val_dataset.mission
        graph_type = self.kwargs["graph_model"]
        # print("Here, we ----do---- calculate again the golden-tcrs")
        train = Repertoires("train", random_sample_from_train)
        file_directory_path = os.path.join("..", "TCR_Dataset2", "Train")  # TCR_Dataset2 exists only in server
        # sample only some sample according to input sample size, and calc the golden tcrs only from them
        train_idx = random.sample(list(train_idx), random_sample_from_train)
        train_files = [Path(os.path.join(file_directory_path, self.train_val_dataset.subject_list[id] + ".csv"))
                       for id in train_idx]
        # print("Length of chosen files", len(train_files))
        numrec = int(self.RECEIVED_PARAMS["numrec"])  # cutoff is also a hyper-parameter
        # print("Number of golden-tcrs", numrec)
        train.save_data(file_directory_path, files=train_files)
        # train.outlier_finder(i, numrec=numrec, cutoff=cutoff)
        # save files' names
        outliers_pickle_name = f"graph_type_{graph_type}_tcr_outliers_{numrec}_with_sample_size_{len(train_files)}_run_number_{i}_mission_{mission}"
        adj_mat_path = f"graph_type_{graph_type}_tcr_mat_{numrec}_with_sample_size_{len(train_files)}_run_number_{i}_mission_{mission}"
        outlier = train.new_outlier_finder(numrec, pickle_name=outliers_pickle_name)  # find outliers and save to pickle
        if graph_type == "projection":
            # create distance matrix between the projection of the found golden tcrs
            # create_distance_matrix(self.device, outliers_file=outliers_pickle_name, adj_mat=adj_mat_path)
            proj_matrix = self.create_projection_tcr_network(outliers_pickle_name, adj_mat_path)
        else:
            corr_df_between_golden_tcrs = self.create_corr_tcr_network(train_idx, file_directory_path, outlier, adj_mat_path)
        # TODO: change back to option
        self.train_val_dataset.run_number = i
        self.test_dataset.run_number = i

        self.train_val_dataset.calc_golden_tcrs(adj_mat_path=adj_mat_path)
        self.train_val_dataset.update_graphs()
        self.test_dataset.calc_golden_tcrs(adj_mat_path=adj_mat_path)
        self.test_dataset.update_graphs()
        return train_idx

    def create_corr_tcr_network(self, train_idx, file_directory_path, outlier, corr_file_name, Threshold=0.2):
        # Here, the graph is created with correlation between the existence of golden tcrs on training set only
        def arrange_corr_between_golden_tcr_mat(corr_df_between_golden_tcrs, Threshold=Threshold):
            corr_df_between_golden_tcrs.values[[np.arange(corr_df_between_golden_tcrs.shape[0])]*2] = 0
            new_corr_df_between_golden_tcrs = (np.abs(corr_df_between_golden_tcrs) >= Threshold).astype(int)
            # new_corr_df_between_golden_tcrs = np.abs(corr_df_between_golden_tcrs)
            return new_corr_df_between_golden_tcrs
        train_samples_golden_tcrs_existence_mat = []
        # golden_tcrs = [i for i, j in list(outlier.keys())]
        golden_tcrs = list(outlier.keys())
        train_subject_list = [self.train_val_dataset.subject_list[id] for id in train_idx]
        for i, subject in tqdm(enumerate(train_subject_list), desc='Create corr matrix tcrs', total=len(train_subject_list)):
            file_path = os.path.join(file_directory_path, subject + ".csv")
            samples_df = pd.read_csv(file_path, usecols=["combined", "frequency"])
            no_rep_sample_df = samples_df.groupby("combined").sum()  # sum the repetitions
            golden_tcr_existence_vector = [0] * len(golden_tcrs)
            cur_sample_tcrs = set(list(no_rep_sample_df.index))
            # Create existence vector of golden tcrs for each training sample
            for inx, golden_tcr in enumerate(golden_tcrs):
                if golden_tcr in cur_sample_tcrs:
                    golden_tcr_existence_vector[inx] = 1
            train_samples_golden_tcrs_existence_mat.append(golden_tcr_existence_vector)
        df = pd.DataFrame(data=train_samples_golden_tcrs_existence_mat, columns=golden_tcrs)
        corr_df_between_golden_tcrs = df.corr(method="spearman")
        threshold = float(self.RECEIVED_PARAMS['thresh'])
        # corr_df_between_golden_tcrs.to_csv(f"{corr_file_name}.csv")

        corr_df_between_golden_tcrs = arrange_corr_between_golden_tcr_mat(corr_df_between_golden_tcrs, Threshold=threshold)
        corr_df_between_golden_tcrs.to_csv(f"{corr_file_name}.csv")
        return corr_df_between_golden_tcrs

    def create_projection_tcr_network(self, outliers_pickle_name, adj_mat_path, Threshold=0.2):
        proj_matrix = create_distance_matrix(self.device, outliers_file=outliers_pickle_name, adj_mat=adj_mat_path)
        # threshold = float(self.RECEIVED_PARAMS['thresh'])
        threshold = 0.62
        df_proj_matrix = pd.DataFrame(proj_matrix)
        df_proj_matrix.set_index(0, inplace=True)
        df_proj_matrix.columns = df_proj_matrix.iloc[0]
        df_proj_matrix.drop(df_proj_matrix.index[0], inplace=True)

        np.fill_diagonal(df_proj_matrix.values, 1)
        df_proj_matrix = 1 / df_proj_matrix
        np.fill_diagonal(df_proj_matrix.values, 0)
        df_proj_matrix = 10 * df_proj_matrix
        # df_proj_matrix.to_csv(f"{adj_mat_path}.csv")
        df_proj_matrix_binary = (np.abs(df_proj_matrix) >= threshold).astype(int)
        df_proj_matrix_binary.to_csv(f"{adj_mat_path}.csv")
        # with open(f"{adj_mat_path}.csv", "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(df_proj_matrix)
        return df_proj_matrix_binary

    def create_data_loaders(self, i, train_idx, val_idx):
        batch_size = int(self.RECEIVED_PARAMS['batch_size'])
        # For Tcr dataset
        if "TCR" in str(self.train_val_dataset):
            train_idx = self.tcr_dataset_dealing(train_idx, i)
            # train_idx = self.compare_network(train_idx, i)
        # Datasets
        train_data = torch.utils.data.Subset(self.train_val_dataset, train_idx)
        print("len of train data", len(train_data))
        val_data = torch.utils.data.Subset(self.train_val_dataset, val_idx)
        print("len of val data", len(val_data))

        # Get and set train_graphs_list
        train_graphs_list = get_train_graphs_list(train_data)
        self.train_val_dataset.set_train_graphs_list(train_graphs_list)
        self.test_dataset.set_train_graphs_list(train_graphs_list)

        # Dataloader
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
        self.train_loader = train_loader
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
        # return None, None, None
