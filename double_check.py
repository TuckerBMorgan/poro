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
linear = nn.Linear(100, 100, bias=True)
adam_optimizer = torch.optim.AdamW(linear.parameters(), lr=0.01)

write_fp32(linear.weight, open("data/tests/adam_optimizer/weight.txt", "wb"))
write_fp32(linear.bias, open("data/tests/adam_optimizer/bias.txt", "wb"))
input = torch.randn(100, 100, dtype=torch.float32)
write_fp32(input, open("data/tests/adam_optimizer/input.txt", "wb"))
fake_output = torch.randn(100, 100, dtype=torch.float32)
write_fp32(fake_output, open("data/tests/adam_optimizer/fake_output.txt", "wb"))

for i in range(0, 10):
    adam_optimizer.zero_grad()
    output = linear(input)
    loss = output - fake_output
    loss = loss.sum()
    loss.backward()
    print(linear.weight.grad)
    adam_optimizer.step()
print(linear.weight)