import torch
import torch.nn as nn
import torch.nn.functional as F


class JustGraphStructure(nn.Module):
    def __init__(self, data_size, RECEIVED_PARAMS):
        super(JustGraphStructure, self).__init__()
        self.data_size = data_size
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        # משקול לפני שמכניסים לרשת
        self.pre_weighting = nn.Linear(self.data_size, self.RECEIVED_PARAMS["preweight"])
        self.fc1 = nn.Linear(self.RECEIVED_PARAMS["preweight"], self.RECEIVED_PARAMS["layer_1"])  # input layer
        self.fc2 = nn.Linear(self.RECEIVED_PARAMS["layer_1"], self.RECEIVED_PARAMS["layer_2"])
        self.fc3 = nn.Linear(self.RECEIVED_PARAMS["layer_2"], 1)
        self.alpha = torch.rand(1, requires_grad=True)

        # self.bn1 = nn.BatchNorm1d(self.RECEIVED_PARAMS["layer_1"])
        # self.bn2 = nn.BatchNorm1d(self.RECEIVED_PARAMS["layer_2"])
        self.activation_func = self.RECEIVED_PARAMS['activation']
        # self.write_to_log()

    def forward(self, x, adjacency_matrix):
        # x = x.view(-1, self.data_size)
        # alpha_A = adjacency_matrix * self.alpha.expand_as(adjacency_matrix)
        alpha_A = torch.mul(adjacency_matrix, self.alpha)
        # alpha_A = torch.matmul(adjacency_matrix, self.alpha.expand_as(adjacency_matrix))
        a, b, c = alpha_A.shape
        d, e = x.shape
        I = torch.eye(b)
        alpha_A_plus_I = alpha_A + I
        x = torch.reshape(x, (d, e, 1))
        x = torch.matmul(alpha_A_plus_I, x)
        x = torch.reshape(x, (d, e))
        x = torch.sign(x)

        if self.activation_func == 'relu':
            x = F.relu(self.pre_weighting(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            # x = self.bn1(F.relu(self.fc1(x)))
            # x = self.bn2(F.relu(self.fc2(x)))
        elif self.activation_func == 'elu':
            x = F.elu(self.pre_weighting(x))
            x = F.elu(self.fc1(x))
            x = F.elu(self.fc2(x))
            # x = self.bn1((F.elu(self.fc1(x))))
            # x = self.bn2((F.elu(self.fc2(x))))
        elif self.activation_func == 'tanh':
            x = torch.tanh(self.pre_weighting(x))
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            # x = self.bn1(torch.tanh(self.fc1(x)))
            # x = self.bn2((torch.tanh(self.fc2(x))))
        # x = torch.sigmoid(x) # BCE loss automatically applies sigmoid
        x = self.fc3(x)
        return x

    def write_to_log(self):
        settings.log_file.write(f"Model name: {self.__class__.__name__}")
        settings.log_file.write(f"Model parameters: {self.RECEIVED_PARAMS}")


# def _train(model, RECEIVED_PARAMS, train_loader, test_loader, loss_weights, device='cpu'):
#     """
#     The training function. We train our model for 10 epochs (like requested). Every epoch we zero
#     the optimizer grad from the earlier epoch, calculate our model, calculate loss, do backpropagation
#     and one optimizing step.
#     """
#     # settings.log_file.write("Training process started..")
#     # for the plots afterwards
#     train_loss_vec = []
#     test_loss_vec = []
#     # train_acc = []
#     # test_acc = []
#     test_auc_vec = []
#     train_auc_vec = []
#     optimizer = get_optimizer(RECEIVED_PARAMS, model)
#     loss = 0
#     epochs = RECEIVED_PARAMS['epochs']
#     # run the main training loop
#     for epoch in range(epochs):
#         # settings.log_file.write(f"Epoch {epoch}")
#         print(f"epoch {epoch}")
#         model.train()
#         for batch_idx, (A, data, target) in enumerate(train_loader):
#             A, data, target = A.to(device), data.to(device), target.to(device)
#             optimizer.zero_grad()
#             net_out = model(data, A)
#             loss = F.binary_cross_entropy_with_logits(net_out, target.unsqueeze(dim=1).float(),
#                                                       weight=torch.Tensor([loss_weights[i] for i in target]).unsqueeze(dim=1).to(device))
#             loss.backward()
#             optimizer.step()
#         # settings.log_file.write(f"Train loss {loss.item()}")
#
#         train_loss_vec.append(loss.item())
#         train_auc, train_loss = _test(model, train_loader, loss_weights, device)
#         print(f"Train AUC: {train_auc}, Train Loss: {train_loss}")
#         # settings.log_file.write(f"Train auc {train_auc}")
#
#         test_auc, test_loss = _test(model, test_loader, loss_weights, device)
#         test_auc_vec.append(test_auc)
#         test_loss_vec.append(test_loss)
#         print(f"Test AUC: {test_auc}, Test Loss: {test_loss}")
#         # settings.log_file.write(f"Test loss {test_loss}")
#         # settings.log_file.write(f"Test auc {test_auc}")
#     # settings.log_file.write("Training process finished..")
#     return train_loss_vec, train_auc_vec, test_loss_vec, test_auc_vec
#
#
# def _binary_acc(y_pred, y_test):
#     y_pred_tag = torch.round(torch.sigmoid(y_pred))
#     correct_results_sum = (y_pred_tag == y_test).sum().float()
#     acc = correct_results_sum / y_test.shape[0]
#     acc = torch.round(acc * 100)
#     return acc
#
#
# def _test(model, set, loss_weights, device, job='Test'):
#     """
#     The test function. for every (data, traget) in our train set we calculate the output according
#     to our model, calculating the loss, get the index of the max log-probability, and claculate
#     "correct" variable to help us later calculate the accuracy.
#     """
#     model.eval()
#     all_targets = []
#     all_pred = []
#     loss = 0
#     for A, data, target in set:
#         data, target = data.to(device), target.to(device)
#         output = model(data, A)
#         if job == 'Test':
#             loss = F.binary_cross_entropy_with_logits(output, target.unsqueeze(dim=1).float(),
#                                                       weight=torch.Tensor([loss_weights[i] for i in target]).unsqueeze(dim=1).to(device))
#             output = torch.sigmoid(output)
#         for i in target:
#             all_targets.append(i.item())
#         output = output.squeeze()
#         for i in output:
#             all_pred.append(i.item())
#     auc_result = roc_auc_score(all_targets, all_pred)
#     return auc_result, loss
#
#
# def get_optimizer(RECEIVED_PARAMS, model):
#     optimizer = RECEIVED_PARAMS['optimizer']
#     learning_rate = RECEIVED_PARAMS['learning_rate']
#     weight_decay = RECEIVED_PARAMS['regularization']
#     if optimizer == 'adam':
#         return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     elif optimizer == 'SGD':
#         return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
