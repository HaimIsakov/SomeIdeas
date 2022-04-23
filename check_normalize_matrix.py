import torch


def calculate_adjacency_matrix(batched_adjacency_matrix):
    # D^(-0.5)
    def calc_d_minus_root_sqr(batched_adjacency_matrix):
        r = []
        for adjacency_matrix in batched_adjacency_matrix:
            print("Cur adjacency_matrix")
            print(adjacency_matrix)
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


if __name__ == "__main__":
    # tensor = torch.ones([2,3,3])
    # normalized_adjacency = calculate_adjacency_matrix(tensor)
    # print(normalized_adjacency)
    alpha = torch.tensor([3])
    I = torch.eye(3).to()
    alpha_I = I * alpha.expand_as(I)  # 𝛼I
    x=1