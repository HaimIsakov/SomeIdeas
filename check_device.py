import torch
device = f"cuda:0" if torch.cuda.is_available() else "cpu"
print("device", device)