import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import optim
import numpy as np
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from EarlyStopping import EarlyStopping

TRAIN_JOB = 'train'
TEST_JOB = 'test'
VAL_JOB = 'validation'
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.75


class TrainTestValOneTime:
    def __init__(self, model, RECEIVED_PARAMS, train_loader, val_loader, test_loader, device, early_stopping=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # self.loss_weights = self.calc_weighs_for_loss()  # according to train loader
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.train_loss_vec, self.val_loss_vec = [], []
        self.train_auc_vec, self.val_auc_vec = [], []
        self.train_acc, self.val_acc = [], []
        self.val_auc = 0.0
        self.early_stopping = early_stopping

    def calc_loss_test(self, data_loader, job=VAL_JOB):
        self.model.eval()
        batched_test_loss = []
        with torch.no_grad():
            for data, adjacency_matrix, target in data_loader:
                data, adjacency_matrix, target = data.to(self.device), adjacency_matrix.to(self.device), target.to(self.device)
                output = self.model(data, adjacency_matrix)
                loss = F.binary_cross_entropy_with_logits(output, target.unsqueeze(dim=1).float())
                # loss = F.binary_cross_entropy_with_logits(output, target.unsqueeze(dim=1).float(),
                #                                           weight=torch.Tensor(
                #                                               [self.loss_weights[i] for i in target]).unsqueeze(
                #                                               dim=1).to(self.device))
                batched_test_loss.append(loss.item())
        average_loss = np.average(batched_test_loss)
        return average_loss

    def calc_auc(self, data_loader, job=VAL_JOB):
        self.model.eval()
        true_labels = []
        pred = []
        with torch.no_grad():
            for data, adjacency_matrix, target in data_loader:
                data, adjacency_matrix, target = data.to(self.device), adjacency_matrix.to(self.device), target.to(self.device)
                output = self.model(data, adjacency_matrix)
                output = torch.sigmoid(output)
                true_labels += target.tolist()
                pred += output.squeeze(dim=1).tolist()
        try:
            auc_result = roc_auc_score(true_labels, pred)
        except:
            print("An Error")
        return auc_result

    def train(self):
        optimizer = self.get_optimizer()
        epochs = int(self.RECEIVED_PARAMS['epochs'])
        # min_val_loss = float('inf')
        # best_model = copy.deepcopy(self.model)
        best_model = self.model.state_dict()
        # max_val_auc = 0.5
        max_val_auc = 0
        counter = 0
        early_training_results = {'val_auc': 0, 'train_auc': 0, 'test_auc': 0}
        # early_training_results['val_auc'] = 0.5
        # early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)
        # run the main training loop
        for epoch in range(epochs):
            self.model.train()  # prep model for training
            batched_train_loss = []
            for data, adjacency_matrix, target in self.train_loader:
                data, adjacency_matrix, target = data.to(self.device), adjacency_matrix.to(self.device), target.to(self.device)
                optimizer.zero_grad()  # clear the gradients of all optimized variables
                net_out = self.model(data, adjacency_matrix)  # forward pass: compute predicted outputs by passing inputs to the model
                loss = F.binary_cross_entropy_with_logits(net_out, target.unsqueeze(dim=1).float())
                # loss = F.binary_cross_entropy_with_logits(net_out, target.unsqueeze(dim=1).float(),
                #                                           weight=torch.Tensor(
                #                                               [self.loss_weights[i] for i in target]).unsqueeze(
                #                                               dim=1).to(self.device))  # calculate the weighted loss
                loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()  # perform a single optimization step (parameter update)
                batched_train_loss.append(loss.item())

            average_train_loss, train_auc, val_loss, val_auc = self.record_evaluations(batched_train_loss)
            # early_training_results['train_auc'] = train_auc
            # if val_auc == 0.5 and train_auc == 0.5:
            #     print("The model stuck")
            if val_auc > max_val_auc:
                print(f"Validation AUC increased ({max_val_auc:.6f} --> {val_auc:.6f})")
                max_val_auc = val_auc
                counter = 0
                early_training_results['val_auc'] = val_auc
                early_training_results['val_loss'] = val_loss
                early_training_results['train_auc'] = train_auc
                # best_model = copy.deepcopy(self.model)
                best_model = self.model.state_dict()
            elif self.early_stopping and counter == EARLY_STOPPING_PATIENCE:
                print("Early stopping")
                #self.model = best_model
                self.model.load_state_dict(best_model)
                early_training_results['test_auc'] = self.calc_auc(self.test_loader, job=TEST_JOB)
                break
            else:
                counter += 1
                print(f'Early-Stopping counter: {counter} out of {EARLY_STOPPING_PATIENCE}')

            # early stopping according to val_loss
            # if val_loss =< min_val_loss:
            #     print(f"Validation loss decreased ({min_val_loss:.6f} --> {val_loss:.6f})")
            #     min_val_loss = val_loss
            #     counter = 0
            #     early_training_results['val_auc'] = val_auc
            #     early_training_results['val_loss'] = val_loss
            #     best_model = copy.deepcopy(self.model)
            # elif self.early_stopping and counter == EARLY_STOPPING_PATIENCE:
            #     print("Early stopping")
            #     self.model = best_model
            #     early_training_results['test_auc'] = self.calc_auc(self.test_loader, job=TEST_JOB)
            #     break
            # else:
            #     counter += 1
            #     print(f'Early-Stopping counter: {counter} out of {EARLY_STOPPING_PATIENCE}')
            # early_stopping(val_loss, self.model)
            print_msg = (f'[{epoch}/{epochs}] ' +
                         f'train_loss: {average_train_loss:.9f} train_auc: {train_auc:.9f} ' +
                         f'valid_loss: {val_loss:.6f} valid_auc: {val_auc:.6f}')
            print(print_msg)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     self.val_auc = self.calc_auc(self.val_loader, job=VAL_JOB)
            #     break
        #self.model = best_model
        self.model.load_state_dict(best_model)
        early_training_results['test_auc'] = self.calc_auc(self.test_loader, job=TEST_JOB)
        return early_training_results

    def record_evaluations(self, batched_train_loss):
        average_train_loss = np.average(batched_train_loss)
        self.train_loss_vec.append(average_train_loss)  # record training loss
        train_auc = self.calc_auc(self.train_loader, TRAIN_JOB)
        self.train_auc_vec.append(train_auc)
        val_loss = self.calc_loss_test(self.val_loader, VAL_JOB)
        val_auc = self.calc_auc(self.val_loader, VAL_JOB)
        self.val_auc_vec.append(val_auc)
        self.val_loss_vec.append(val_loss)
        return average_train_loss, train_auc, val_loss, val_auc

    def get_optimizer(self):
        optimizer_name = self.RECEIVED_PARAMS['optimizer']
        learning_rate = self.RECEIVED_PARAMS['learning_rate']
        weight_decay = self.RECEIVED_PARAMS['regularization']
        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=SCHEDULER_PATIENCE, verbose=True, factor=SCHEDULER_FACTOR)
        return optimizer

    def train_geometric(self):
        optimizer = self.get_optimizer()
        epochs = self.RECEIVED_PARAMS['epochs']
        # min_val_loss = float('inf')
        # best_model = copy.deepcopy(self.model)
        best_model = self.model.state_dict()
        # max_val_auc = 0.5
        max_val_auc = 0
        counter = 0
        early_training_results = {}
        # early_training_results['val_auc'] = 0.5
        # early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)
        # run the main training loop
        for epoch in range(epochs):
            self.model.train()  # prep model for training
            batched_train_loss = []
            for data in self.train_loader:
                out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
                loss = F.binary_cross_entropy_with_logits(out, data.y.unsqueeze(dim=1).float())  # Compute the loss.
                optimizer.zero_grad()  # clear the gradients of all optimized variables
                loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()  # perform a single optimization step (parameter update)
                batched_train_loss.append(loss.item())

            average_train_loss, train_auc, val_loss, val_auc = self.record_evaluations_geometric(batched_train_loss)
            # early_training_results['train_auc'] = train_auc
            if val_auc > max_val_auc:
                print(f"Validation AUC increased ({max_val_auc:.6f} --> {val_auc:.6f})")
                max_val_auc = val_auc
                counter = 0
                early_training_results['val_auc'] = val_auc
                early_training_results['val_loss'] = val_loss
                early_training_results['train_auc'] = train_auc
                # best_model = copy.deepcopy(self.model)
                best_model = self.model.state_dict()
            elif self.early_stopping and counter == EARLY_STOPPING_PATIENCE:
                print("Early stopping")
                # self.model = best_model
                self.model.load_state_dict(best_model)
                early_training_results['test_auc'] = self.calc_auc_geometric(self.test_loader, job=TEST_JOB)
                break
            else:
                counter += 1
                print(f'Early-Stopping counter: {counter} out of {EARLY_STOPPING_PATIENCE}')

            # early stopping according to val_loss
            # if val_loss =< min_val_loss:
            #     print(f"Validation loss decreased ({min_val_loss:.6f} --> {val_loss:.6f})")
            #     min_val_loss = val_loss
            #     counter = 0
            #     early_training_results['val_auc'] = val_auc
            #     early_training_results['val_loss'] = val_loss
            #     best_model = copy.deepcopy(self.model)
            # elif self.early_stopping and counter == EARLY_STOPPING_PATIENCE:
            #     print("Early stopping")
            #     self.model = best_model
            #     early_training_results['test_auc'] = self.calc_auc(self.test_loader, job=TEST_JOB)
            #     break
            # else:
            #     counter += 1
            #     print(f'Early-Stopping counter: {counter} out of {EARLY_STOPPING_PATIENCE}')
            # early_stopping(val_loss, self.model)
            print_msg = (f'[{epoch}/{epochs}] ' +
                         f'train_loss: {average_train_loss:.9f} train_auc: {train_auc:.9f} ' +
                         f'valid_loss: {val_loss:.6f} valid_auc: {val_auc:.6f}')
            print(print_msg)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     self.val_auc = self.calc_auc(self.val_loader, job=VAL_JOB)
            #     break
        # self.model = best_model
        self.model.load_state_dict(best_model)
        early_training_results['test_auc'] = self.calc_auc_geometric(self.test_loader, job=TEST_JOB)
        return early_training_results

    def record_evaluations_geometric(self, batched_train_loss):
        average_train_loss = np.average(batched_train_loss)
        self.train_loss_vec.append(average_train_loss)  # record training loss
        train_auc = self.calc_auc_geometric(self.train_loader, TRAIN_JOB)
        self.train_auc_vec.append(train_auc)
        val_loss = self.calc_loss_test_geometric(self.val_loader, VAL_JOB)
        val_auc = self.calc_auc_geometric(self.val_loader, VAL_JOB)
        self.val_auc_vec.append(val_auc)
        self.val_loss_vec.append(val_loss)
        return average_train_loss, train_auc, val_loss, val_auc

    def calc_loss_test_geometric(self, data_loader, job=VAL_JOB):
        self.model.eval()
        batched_test_loss = []
        with torch.no_grad():
            for data in data_loader:
                out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
                loss = F.binary_cross_entropy_with_logits(out, data.y.unsqueeze(dim=1).float())  # Compute the loss.
                batched_test_loss.append(loss.item())
        average_loss = np.average(batched_test_loss)
        return average_loss

    def calc_auc_geometric(self, data_loader, job=VAL_JOB):
        self.model.eval()
        true_labels = []
        pred = []
        with torch.no_grad():
            for data in data_loader:
                out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
                output = torch.sigmoid(out)
                true_labels += data.y.tolist()
                pred += output.squeeze(dim=1).tolist()
        auc_result = roc_auc_score(true_labels, pred)
        return auc_result

    #
    # def calc_weighs_for_loss(self):
    #     count_ones = 0
    #     count_zeros = 0
    #     for batch_index, (A, data, target) in enumerate(self.train_loader):
    #         for i in target:
    #             if i.item() == 1:
    #                 count_ones += 1
    #             if i.item() == 0:
    #                 count_zeros += 1
    #     return [1 / count_zeros, 1 / count_ones]
