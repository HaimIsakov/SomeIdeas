import torch
import torch.nn as nn


class FielderVector(nn.Module):
    def __init__(self, nodes_number, feature_size, RECEIVED_PARAMS, device, num_classes=1):
        super(FielderVector, self).__init__()
        self.feature_size = feature_size
        self.nodes_number = nodes_number
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS

        self.activation_func = self.RECEIVED_PARAMS['activation']
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])

        self.activation_func_dict = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}
        if self.feature_size > 1:
            self.transform_mat_to_vec = nn.Linear(self.feature_size, 1)

        self.values_classifier = nn.Sequential(
            nn.Linear(1, int(self.RECEIVED_PARAMS["values_layer_1"])),
            nn.BatchNorm1d(self.nodes_number),
            self.activation_func_dict[self.activation_func],
            nn.Dropout(p=self.RECEIVED_PARAMS["dropout"]),
            nn.Linear(int(self.RECEIVED_PARAMS["values_layer_1"]), int(self.RECEIVED_PARAMS["values_layer_2"])),
            self.activation_func_dict[self.activation_func]
        )
        self.fiedler_vector_classifier = nn.Sequential(
            nn.Linear(1, int(self.RECEIVED_PARAMS["fiedler_layer_1"])),
            nn.BatchNorm1d(self.nodes_number),
            self.activation_func_dict[self.activation_func],
            nn.Dropout(p=self.RECEIVED_PARAMS["dropout"]),
            nn.Linear(int(self.RECEIVED_PARAMS["fiedler_layer_1"]), int(self.RECEIVED_PARAMS["fiedler_layer_2"])),
            self.activation_func_dict[self.activation_func]
        )
        self.fc3 = nn.Linear(int(self.RECEIVED_PARAMS["fiedler_layer_2"]) * self.nodes_number +
                             int(self.RECEIVED_PARAMS["values_layer_2"]) * self.nodes_number, num_classes)

    def forward(self, x, fiedler_vector):
        values_embedding = self.values_classifier(x)
        values_embedding = torch.flatten(values_embedding, start_dim=1)  # flatten the tensor

        fiedler_vector_embedding = self.fiedler_vector_classifier(fiedler_vector)
        fiedler_vector_embedding = torch.flatten(fiedler_vector_embedding, start_dim=1)  # flatten the tensor

        concat_graph_and_values = torch.cat((values_embedding, fiedler_vector_embedding), 1)

        values_fiedler_concat_embedding = self.fc3(concat_graph_and_values)
        #  Consider adding LayerNorm or BatchNorm
        return values_fiedler_concat_embedding
