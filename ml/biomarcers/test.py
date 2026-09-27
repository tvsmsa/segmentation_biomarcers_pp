import torch

state_dict = torch.load('D:/models/deeplab_tversky/fold_1.pth')
adapted_dict = {}
print(state_dict)