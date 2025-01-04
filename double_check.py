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
test_input.requires_grad = True
test_target = torch.randn(4, 64, 768)
x_shape = test_input.shape
mean_indices = []
mean = test_input.mean(1, keepdim=True)
input_minus_mean = test_input - mean
input_var = (input_minus_mean * input_minus_mean).mean(mean_indices)
print(input_var)
exit()

#for index in x_shape:


exit()

output = layer_norm(test_input)

loss = F.mse_loss(output, test_target)
loss.backward()
'''
layer_norm_weights_path = "data/tests/layer_norm/layer_norm_weights.txt"
layer_norm_bias_path = "data/tests/layer_norm/layer_norm_bias.txt"
test_input_path = "data/tests/layer_norm/test_input.txt"
expected_output_path = "data/tests/layer_norm/expected_output.txt"
fake_target = "data/tests/layer_norm/fake_target.txt"
expected_loss = "data/tests/layer_norm/expected_loss.txt"
layer_norm_weights_grad = "data/tests/layer_norm/layer_norm_weights_grad.txt"
layer_norm_bias_grad = "data/tests/layer_norm/layer_norm_bias_grad.txt"

write_fp32(layer_norm.weight, open(layer_norm_weights_path, "wb"))
write_fp32(layer_norm.bias, open(layer_norm_bias_path, "wb"))
write_fp32(test_input, open(test_input_path, "wb"))
write_fp32(output, open(expected_output_path, "wb"))
write_fp32(test_target, open(fake_target, "wb"))
write_fp32(torch.Tensor(loss), open(expected_loss, "wb"))
write_fp32(layer_norm.weight.grad, open(layer_norm_weights_grad, "wb"))
write_fp32(layer_norm.bias.grad, open(layer_norm_bias_grad, "wb"))
'''
print(test_input.shape)
'''
files = []
files.append(open("./data/tests/causal_self_attention/causal_self_attention_value_bias_grad.txt", "wb"))

write_fp32(causal_self_attention.v.bias.grad, files[19])
'''