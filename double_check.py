import math
import torch


# I need two tensors, one of shape [3, 2, 2] and one of [2, 2]

# Create the first tensor
tensor1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

# Create the second tensor
tensor2 = torch.tensor([[1, 2], [3, 4]])

a = tensor1 @ tensor2
print(a.shape)