import copy

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import optim
import numpy as np

TRAIN_JOB = 'train'
TEST_JOB = 'test'
VAL_JOB = 'validation'
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.75
# all_models_output = []
# all_real_tags = []


class TrainTestValOneTime:
    def __init__(self, model, RECEIVED_PARAMS, train_loader, val_loader, test_loader, device, num_classes=1, early_stopping=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.train_loss_vec, self.val_loss_vec = [], []
        self.train_auc_vec, self.val_auc_vec = [], []
        self.train_acc, self.val_acc = [], []
        self.alpha_list = []
        self.val_auc = 0.0
        self.early_stopping = early_stopping
        self.num_classes = num_classes

    def loss(self, output, target):
        if self.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(output, target.unsqueeze(dim=1).float())
        return loss

    def calc_loss_test(self, data_loader, job=VAL_JOB):
        running_loss = 0
        self.model.eval()
        batched_test_loss = []
        with torch.no_grad():
            for data, adjacency_matrix, target in data_loader:
                data, adjacency_matrix, target = data.to(self.device), adjacency_matrix.to(self.device), target.to(self.device)
                output = self.model(data, adjacency_matrix)
                loss = self.loss(output, target)
                running_loss += loss.item()
                batched_test_loss.append(loss.item())
        average_running_loss = running_loss / len(data_loader.dataset)
        return average_running_loss

    def calc_auc(self, data_loader, job=VAL_JOB):
        self.model.eval()
        true_labels = []
        pred = []
        with torch.no_grad():
            for data, adjacency_matrix, target in data_loader:
                data, adjacency_matrix, target = data.to(self.device), adjacency_matrix.to(self.device), \
                                                 target.to(self.device)
                output = self.model(data, adjacency_matrix)
                true_labels += target.tolist()
                if self.num_classes == 1:
                    output = torch.sigmoid(output)
                    pred += output.squeeze(dim=1).tolist()
        if self.num_classes == 1:
            # for the calculation of auc on all
            # all_models_output.append(pred)
            # all_real_tags.append(true_labels)
            # For the regular auc calculation
            try:
                metric_result = roc_auc_score(true_labels, pred)
            except Exception as e:
                metric_result = 0.5
                print(e)
        return metric_result

    def pred_for_all_auc(self, data_loader, all_models_output, all_real_tags, job=VAL_JOB):
        self.model.eval()
        true_labels = []
        pred = []
        with torch.no_grad():
            for data, adjacency_matrix, target in data_loader:
                data, adjacency_matrix, target = data.to(self.device), adjacency_matrix.to(self.device), \
                                                 target.to(self.device)
                output = self.model(data, adjacency_matrix)
                true_labels += target.tolist()
                output = torch.sigmoid(output)
                pred += output.squeeze(dim=1).tolist()
                # for the calculation of auc on all
                all_models_output.append(pred)
                all_real_tags.append(true_labels)


    def train(self):
        optimizer = self.get_optimizer()
        epochs = 200
        best_model = copy.deepcopy(self.model)
        max_val_auc = 0
        counter = 0

        early_training_results = {'val_auc': 0, 'train_auc': 0, 'test_auc': 0}
        # run the main training loop
        for epoch in range(epochs):
            running_loss = 0

            self.model.train()  # prep model for training
            batched_train_loss = []
            for data, adjacency_matrix, target in self.train_loader:
                data, adjacency_matrix, target = data.to(self.device), adjacency_matrix.to(self.device), target.to(self.device)
                optimizer.zero_grad()  # clear the gradients of all optimized variables
                net_out = self.model(data, adjacency_matrix)  # forward pass: compute predicted outputs by passing inputs to the model
                loss = self.loss(net_out, target)
                running_loss += loss.item() * data.size(0)
                loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()  # perform a single optimization step (parameter update)
                batched_train_loss.append(loss.item())
            try:
                self.alpha_list.append(self.model.alpha.item())
                print("Alpha value:", self.model.alpha.item())
            except:
                pass
            average_train_loss, train_auc, val_loss, val_auc = self.record_evaluations(batched_train_loss)
            average_running_loss = running_loss / len(self.train_loader.dataset)

            test_auc = self.calc_auc(self.test_loader, job=TEST_JOB)

            print("Early-stopping according to validation auc")
            if val_auc > max_val_auc:
                print(f"Validation AUC increased ({max_val_auc:.6f} --> {val_auc:.6f})")
                max_val_auc = val_auc
                counter = 0
                early_training_results['val_auc'], early_training_results['val_loss'] = val_auc, val_loss
                early_training_results['train_auc'], early_training_results['train_loss'] = train_auc, average_running_loss
                best_model = copy.deepcopy(self.model)
            elif self.early_stopping and counter == EARLY_STOPPING_PATIENCE:
                print("Early stopping")
                self.model = best_model
                early_training_results['test_auc'] = self.calc_auc(self.test_loader, job=TEST_JOB)
                early_training_results['last_alpha_value'] = self.get_alpha_value()
                # break
                return early_training_results
            else:
                counter += 1
                print(f'Early-Stopping counter: {counter} out of {EARLY_STOPPING_PATIENCE}')
            ########################
            # print("Early-stopping according to loss")
            # if val_loss <= min_val_loss:
            #     print(f"Validation loss decreased ({min_val_loss:.6f} --> {val_loss:.6f})")
            #     min_val_loss = val_loss
            #     counter = 0
            #     early_training_results['val_auc'], early_training_results['val_loss'] = val_auc, val_loss
            #     early_training_results['train_auc'], early_training_results['train_loss'] = train_auc, average_train_loss
            #     best_model = copy.deepcopy(self.model)
            #     # best_model = self.model.state_dict()
            #
            # elif self.early_stopping and counter == EARLY_STOPPING_PATIENCE:
            #     print("Early stopping")
            #     self.model = best_model
            #     early_training_results['test_auc'] = self.calc_auc(self.test_loader, job=TEST_JOB)
            #     early_training_results['last_alpha_value'] = self.get_alpha_value()
            #     # break
            #     return early_training_results
            # else:
            #     counter += 1
            #     print(f'Early-Stopping counter: {counter} out of {EARLY_STOPPING_PATIENCE}')

            print_msg = (f'[{epoch}/{epochs}] ' +
                         f'train_loss: {average_running_loss:.9f} train_auc: {train_auc:.9f} ' +
                         f'valid_loss: {val_loss:.6f} valid_auc: {val_auc:.6f} ' +
                         f'test_auc: {test_auc:.6f}')
            print(print_msg)
        # calculate test_auc if the model run for all epochs (i.e.: early stopping did not occur)
        # self.model.load_state_dict(best_model)
        self.model = best_model
        early_training_results['test_auc'] = self.calc_auc(self.test_loader, job=TEST_JOB)
        early_training_results['last_alpha_value'] = self.get_alpha_value()
        # early_training_results = self.calc_auc_from_all_comparison(early_training_results)
        return early_training_results

    def get_alpha_value(self):
        try:
            alpha_value = self.model.alpha.item()
            print("Alpha_value in get_alpha_value function", alpha_value)
            return alpha_value
        except:
            pass

    def calc_auc_from_all_comparison(self, early_training_results):
        early_training_results['test_auc'] = self.calc_auc(self.test_loader, job=TEST_JOB)
        real_tags = np.ravel(np.array(all_real_tags))
        models_output = np.ravel(np.array(all_models_output))
        print("all_real_tags", real_tags)
        print("all_models_output", models_output)
        test_auc_from_all = roc_auc_score(real_tags, models_output)
        print("--------------------------------------------------------------------------")
        print(f"test auc from all comparisons {test_auc_from_all:.6f}")
        print("--------------------------------------------------------------------------")
        early_training_results['all_test_together'] = test_auc_from_all
        return early_training_results

    def record_evaluations(self, batched_train_loss):
        average_train_loss = np.average(batched_train_loss)
        self.train_loss_vec.append(average_train_loss)  # record training loss
        train_auc = self.calc_auc(self.train_loader, TRAIN_JOB)
        val_auc = self.calc_auc(self.val_loader, VAL_JOB)
        self.train_auc_vec.append(train_auc)
        val_loss = self.calc_loss_test(self.val_loader, VAL_JOB)
        self.val_auc_vec.append(val_auc)
        self.val_loss_vec.append(val_loss)
        return average_train_loss, train_auc, val_loss, val_auc

    def get_optimizer(self):
        learning_rate = self.RECEIVED_PARAMS['learning_rate']
        weight_decay = self.RECEIVED_PARAMS['regularization']
        if 'optimizer' in self.RECEIVED_PARAMS:
            optimizer_name = self.RECEIVED_PARAMS['optimizer']
            if optimizer_name == 'adam':
                optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_name == 'SGD':
                optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        return optimizer
