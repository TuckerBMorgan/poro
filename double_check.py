import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import math
def write_floats_to_file(path: str, data) -> None:
    with open(path, 'a') as file:
        for value in data:
            file.write(f"{value}\n")

def append_string_to_file(path: str, content: str) -> None:
    with open(path, 'a') as file:
        file.write(f"{content}\n")

def write_fp32(tensor, file):
    # first write the length of the tensor's shape
    shape = torch.tensor(tensor.size(), dtype=torch.int32)
    # write the number of dimensions
    file.write(struct.pack("<i", len(shape)))
    file.write(shape.numpy().tobytes())
    # then write the tensor's shape
    # then write the tensor's data
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)



torch.set_printoptions(precision=8)
torch.manual_seed(42)
layer_norm = nn.LayerNorm(768)
test_input = torch.randn(4, 64, 768)
output = layer_norm(test_input)


'''
files = []
files.append(open("./data/tests/causal_self_attention/causal_self_attention_value_bias_grad.txt", "wb"))

write_fp32(causal_self_attention.v.bias.grad, files[19])
'''