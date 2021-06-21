import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import optim

TRAIN_JOB = 'train'
TEST_JOB = 'test'

class TrainTestValOneTime:
    def __init__(self, model, RECEIVED_PARAMS, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_weights = self.calc_weighs_for_loss()  # according to train loader
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.train_loss_vec, self.test_loss_vec = [], []
        self.train_auc_vec, self.test_auc_vec = [], []
        self.train_acc, self.test_acc = [], []

        self.train()

    def test(self, data_loader, job=TEST_JOB):
        self.model.eval()
        all_targets = []
        all_pred = []
        loss = 0
        for A, data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data, A)
            if job == TEST_JOB:
                loss = F.binary_cross_entropy_with_logits(output, target.unsqueeze(dim=1).float(),
                                                          weight=torch.Tensor(
                                                              [self.loss_weights[i] for i in target]).unsqueeze(
                                                              dim=1).to(self.device))
                output = torch.sigmoid(output)
            for i in target:
                all_targets.append(i.item())
            output = output.squeeze()
            for i in output:
                all_pred.append(i.item())
        auc_result = roc_auc_score(all_targets, all_pred)
        return auc_result, loss

    def train(self):
        optimizer = self.get_optimizer()
        epochs = self.RECEIVED_PARAMS['epochs']
        loss = 0
        # run the main training loop
        for epoch in range(epochs):
            print(f"epoch {epoch}")
            self.model.train()
            for batch_idx, (A, data, target) in enumerate(self.train_loader):
                A, data, target = A.to(self.device), data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                net_out = self.model(data, A)
                loss = F.binary_cross_entropy_with_logits(net_out, target.unsqueeze(dim=1).float(),
                                                          weight=torch.Tensor(
                                                              [self.loss_weights[i] for i in target]).unsqueeze(
                                                              dim=1).to(self.device))
                loss.backward()
                optimizer.step()
            self.train_loss_vec.append(loss.item())
            train_auc, train_loss = self.test(self.train_loader, TRAIN_JOB)
            self.train_auc_vec.append(train_auc)
            print(f"Train AUC: {train_auc}, Train Loss: {train_loss}")
            test_auc, test_loss = self.test(self.test_loader, TEST_JOB)
            self.test_auc_vec.append(test_auc)
            self.test_loss_vec.append(test_loss)
            print(f"Test AUC: {test_auc}, Test Loss: {test_loss}")

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
