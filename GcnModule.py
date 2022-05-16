import torch
import torch.nn as nn

# class Alpha(nn.Module):
#     def __init__(self):
#         '''
#         Init method.
#         '''
#         super().__init__() # init the base class
#
#     def forward(self, input):
#         '''
#         Forward pass of the function.
#         '''
#         return srss_func(input) # simply apply already implemented SiLU


class AlphaGcn(nn.Module):
    def __init__(self, feature_size, RECEIVED_PARAMS, device):
        super(AlphaGcn, self).__init__()
        self.feature_size = feature_size
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.min_value = torch.tensor([1e-10], device=device).float()
        self.pre_weighting = nn.Linear(feature_size, int(self.RECEIVED_PARAMS["preweight"]))

        self.alpha = nn.Parameter(torch.rand(1, requires_grad=True, device=self.device))

        self.gcn_layer = nn.Sequential(
            self.pre_weighting,
        )

    def forward(self, x, adjacency_matrix):
        a, b, c = adjacency_matrix.shape
        d, e, f = x.shape
        I = torch.eye(b).to(self.device)
        if self.alpha.item() < self.min_value.item():
            print("In min_value")
            print("alpha value", self.alpha.item(), "min_value", self.min_value.item())
            alpha_I = I * self.min_value.expand_as(I)  # min_value * I
        else:
            alpha_I = I * self.alpha.expand_as(I)  # 𝛼I
        normalized_adjacency_matrix = self.calculate_adjacency_matrix_old(adjacency_matrix)  # Ã
        alpha_I_plus_normalized_A = alpha_I + normalized_adjacency_matrix  # 𝛼I + Ã̃
        x = torch.matmul(alpha_I_plus_normalized_A, x)  # (𝛼I + Ã̃)·x
        x = self.gcn_layer(x)
        return x

    def calculate_adjacency_matrix(self, batched_adjacency_matrix):
        # Here we normalize (𝛼I + A)
        # D^(-0.5)
        def calc_d_minus_root_sqr(batched_adjacency_matrix):
            r = []
            for adjacency_matrix in batched_adjacency_matrix:
                sum_of_each_row = adjacency_matrix.sum(1)
                # sum_of_each_row_plus_one = torch.where(sum_of_each_row != 0, sum_of_each_row, torch.tensor(1.0))
                try:
                    r.append(torch.diag(torch.pow(sum_of_each_row, -0.5)))
                except Exception as e:
                    print(e)
                    raise
            s = torch.stack(r)
            # if torch.isnan(s).any():
            #     print("Alpha when stuck", alpha.item())
            #     print("batched_adjacency_matrix", torch.isnan(batched_adjacency_matrix).any())
            #     print("The model is stuck", torch.isnan(s).any())
            return s

        D__minus_sqrt = calc_d_minus_root_sqr(batched_adjacency_matrix)
        normalized_adjacency = torch.matmul(torch.matmul(D__minus_sqrt, batched_adjacency_matrix), D__minus_sqrt)
        return normalized_adjacency

    def calculate_adjacency_matrix_old(self, batched_adjacency_matrix):
        # Here we normalize only A -> D^(-0.5)*A*D^(-0.5)
        # D^(-0.5)
        def calc_d_minus_root_sqr(batched_adjacency_matrix):
            r = []
            for adjacency_matrix in batched_adjacency_matrix:
                sum_of_each_row = adjacency_matrix.sum(1)
                sum_of_each_row_plus_one = torch.where(sum_of_each_row != 0, sum_of_each_row, torch.tensor(1.0, device=self.device))
                r.append(torch.diag(torch.pow(sum_of_each_row_plus_one, -0.5)))
            s = torch.stack(r)
            if torch.isnan(s).any():
                print("Alpha when stuck", self.alpha.item())
                print("batched_adjacency_matrix", torch.isnan(batched_adjacency_matrix).any())
                print("The model is stuck", torch.isnan(s).any())
            return s
        D__minus_sqrt = calc_d_minus_root_sqr(batched_adjacency_matrix)
        normalized_adjacency = torch.matmul(torch.matmul(D__minus_sqrt, batched_adjacency_matrix), D__minus_sqrt)
        return normalized_adjacency
