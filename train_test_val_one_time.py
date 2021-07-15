import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import optim
import numpy as np
from EarlyStopping import EarlyStopping

TRAIN_JOB = 'train'
TEST_JOB = 'test'
VAL_JOB = 'validation'
PATIENCE = 20


class TrainTestValOneTime:
    def __init__(self, model, RECEIVED_PARAMS, train_loader, val_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # self.loss_weights = self.calc_weighs_for_loss()  # according to train loader
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.train_loss_vec, self.test_loss_vec = [], []
        self.train_auc_vec, self.test_auc_vec = [], []
        self.train_acc, self.test_acc = [], []
        self.test_auc = 0.0
        # self.node_order = node_order
        # self.train()

    def calc_loss_test(self, data_loader, job=VAL_JOB):
        ######################
        # calc loss on val/test set #
        ######################
        self.model.eval()
        batched_test_loss = []
        for A, data, target in data_loader:
            A, data, target = A.to(self.device), data.to(self.device), target.to(self.device)
            # normalized_A = torch.Tensor(normalize_adjacency(gnx, self.node_order))
            output = self.model(data, A)
            loss = F.binary_cross_entropy_with_logits(output, target.unsqueeze(dim=1).float())
            # loss = F.binary_cross_entropy_with_logits(output, target.unsqueeze(dim=1).float(),
            #                                           weight=torch.Tensor(
            #                                               [self.loss_weights[i] for i in target]).unsqueeze(
            #                                               dim=1).to(self.device))
            batched_test_loss.append(loss.item())
        average_loss = np.average(batched_test_loss)
        return average_loss

    def calc_auc(self, data_loader, job=VAL_JOB):
        ######################
        # calculate auc #
        ######################
        self.model.eval()
        all_targets, all_pred = [], []
        for A, data, target in data_loader:
            A, data, target = A.to(self.device), data.to(self.device), target.to(self.device)
            output = self.model(data, A)
            output = torch.sigmoid(output)
            for i in target:
                all_targets.append(i.item())
            output = output.squeeze()
            for i in output:
                all_pred.append(i.item())
        auc_result = roc_auc_score(all_targets, all_pred)
        return auc_result

    def train(self):
        optimizer = self.get_optimizer()
        epochs = self.RECEIVED_PARAMS['epochs']
        early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)
        # run the main training loop
        for epoch in range(epochs):
            ###################
            # train the model #
            ###################
            self.model.train()  # prep model for training
            batched_train_loss = []
            for (A, data, target) in self.train_loader:
                A, data, target = A.to(self.device), data.to(self.device), target.to(self.device)
                optimizer.zero_grad()  # clear the gradients of all optimized variables
                net_out = self.model(data, A)  # forward pass: compute predicted outputs by passing inputs to the model
                loss = F.binary_cross_entropy_with_logits(net_out, target.unsqueeze(dim=1).float())
                # loss = F.binary_cross_entropy_with_logits(net_out, target.unsqueeze(dim=1).float(),
                #                                           weight=torch.Tensor(
                #                                               [self.loss_weights[i] for i in target]).unsqueeze(
                #                                               dim=1).to(self.device))  # calculate the weighted loss
                loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()  # perform a single optimization step (parameter update)
                batched_train_loss.append(loss.item())
            average_train_loss = np.average(batched_train_loss)
            self.train_loss_vec.append(average_train_loss)  # record training loss
            train_auc = self.calc_auc(self.train_loader, TRAIN_JOB)
            self.train_auc_vec.append(train_auc)
            val_loss = self.calc_loss_test(self.val_loader, VAL_JOB)
            val_auc = self.calc_auc(self.val_loader, VAL_JOB)
            self.test_auc_vec.append(val_auc)
            self.test_loss_vec.append(val_loss)
            early_stopping(val_loss, self.model)
            print_msg = (f'[{epoch}/{epochs}] ' +
                         f'train_loss: {average_train_loss:.9f} train_auc: {train_auc:.9f} ' +
                         f'valid_loss: {val_loss:.6f} valid_auc: {val_auc:.6f}')
            if early_stopping.early_stop:
                print("Early stopping")
                self.test_auc = self.calc_auc(self.val_loader, job=VAL_JOB)
                break
            print(print_msg)
        self.test_auc = self.calc_auc(self.val_loader, job=VAL_JOB)

    def get_optimizer(self):
            optimizer = self.RECEIVED_PARAMS['optimizer']
            learning_rate = self.RECEIVED_PARAMS['learning_rate']
            weight_decay = self.RECEIVED_PARAMS['regularization']
            if optimizer == 'adam':
                return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer == 'SGD':
                return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def calc_weighs_for_loss(self):
        count_ones = 0
        count_zeros = 0
        for batch_index, (A, data, target) in enumerate(self.train_loader):
            for i in target:
                if i.item() == 1:
                    count_ones += 1
                if i.item() == 0:
                    count_zeros += 1
        return [1 / count_zeros, 1 / count_ones]
