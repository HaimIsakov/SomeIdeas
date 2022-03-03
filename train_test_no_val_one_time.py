import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from torch import optim
import numpy as np

TRAIN_JOB = 'train'
TEST_JOB = 'test'
VAL_JOB = 'validation'
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.75
all_models_output = []
all_real_tags = []


def get_global_var():
    return {"all_models_output": all_models_output,
            "all_real_tags": all_real_tags}

class TrainTestValOneTimeNoValidation:
    def __init__(self, model, RECEIVED_PARAMS, train_loader, test_loader, device, num_classes=1, early_stopping=True):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.train_loss_vec = []
        self.train_auc_vec = []
        self.train_acc, self.val_acc = [], []
        self.alpha_list = []
        self.early_stopping = early_stopping
        self.num_classes = num_classes if str(self.train_loader) != "cancer" else 34

    def loss(self, output, target):
        if self.num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(output, target.unsqueeze(dim=1).float())
        else:
            loss = F.cross_entropy(output, target.unsqueeze(dim=1).float())
        return loss

    def calc_loss_test(self, data_loader, job=VAL_JOB):
        self.model.eval()
        batched_test_loss = []
        with torch.no_grad():
            for data, adjacency_matrix, target in data_loader:
                data, adjacency_matrix, target = data.to(self.device), adjacency_matrix.to(self.device), \
                                                 target.to(self.device)
                output = self.model(data, adjacency_matrix)
                loss = self.loss(output, target)
                batched_test_loss.append(loss.item())
        average_loss = np.average(batched_test_loss)
        return average_loss

    def calc_auc(self, data_loader, job=VAL_JOB):
        global all_models_output
        global all_real_tags

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
                else:
                    output = F.softmax(output, dim=1)
                    _, y_pred_tags = torch.max(output, dim=1)
                    pred += y_pred_tags.tolist()
        if self.num_classes == 1:
            # for the calculation of auc on all
            if job == TEST_JOB:
                all_models_output.append(pred)
                all_real_tags.append(true_labels)
            metric_result = roc_auc_score(true_labels, pred)
        else:
            metric_result = f1_score(true_labels, pred, average='macro')
        return metric_result

    def train(self):
        optimizer = self.get_optimizer()
        epochs = 50
        counter = 0
        early_training_results = {'train_auc': 0, 'test_auc': 0, "all_test_together": 0}
        # run the main training loop
        for epoch in range(epochs):
            self.model.train()  # prep model for training
            batched_train_loss = []
            for data, adjacency_matrix, target in self.train_loader:
                data, adjacency_matrix, target = data.to(self.device), adjacency_matrix.to(self.device), target.to(self.device)
                optimizer.zero_grad()  # clear the gradients of all optimized variables
                net_out = self.model(data, adjacency_matrix)  # forward pass: compute predicted outputs by passing inputs to the model
                # loss = F.binary_cross_entropy_with_logits(net_out, target.unsqueeze(dim=1).float())
                loss = self.loss(net_out, target)
                loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()  # perform a single optimization step (parameter update)
                batched_train_loss.append(loss.item())
            try:
                self.alpha_list.append(self.model.alpha.item())
                print("Alpha value:", self.model.alpha.item())
            except:
                pass
            average_train_loss, train_auc = self.record_evaluations(batched_train_loss)
            early_training_results['train_auc'], early_training_results['train_loss'] = train_auc, average_train_loss
            print_msg = (f'[{epoch}/{epochs}] ' +
                         f'train_loss: {average_train_loss:.9f} train_auc: {train_auc:.9f}')
            print(print_msg)
        early_training_results['test_auc'] = self.calc_auc(self.test_loader, job=TEST_JOB)
        # early_training_results = self.calc_auc_from_all_comparison(early_training_results)
        return early_training_results


    def record_evaluations(self, batched_train_loss):
        average_train_loss = np.average(batched_train_loss)
        self.train_loss_vec.append(average_train_loss)  # record training loss
        train_auc = self.calc_auc(self.train_loader, TRAIN_JOB)
        self.train_auc_vec.append(train_auc)
        return average_train_loss, train_auc

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
        # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=SCHEDULER_PATIENCE, verbose=True, factor=SCHEDULER_FACTOR)
        return optimizer
