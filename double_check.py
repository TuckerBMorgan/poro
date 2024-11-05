import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import math


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
test_input = torch.rand(2, 768)
fake_target = torch.rand(2, 768)

test_input_mean = test_input.mean(dim=-1, keepdim=True)
print(test_input_mean)

input_minus_mean = test_input - test_input_mean
input_minus_mean_squared = input_minus_mean ** 2
input_minus_mean_squared_mean = input_minus_mean_squared.mean(dim=-1, keepdim=True)
print(input_minus_mean_squared_mean)
exit()

input_minus_mean_squared_mean_sqrt = (input_minus_mean_squared_mean + layer_norm.eps).pow(0.5)

output = input_minus_mean / input_minus_mean_squared_mean_sqrt
output = output * layer_norm.weight + layer_norm.bias
output = layer_norm(test_input)
diff = output - fake_target
print(diff.pow(2).mean())
loss = F.mse_loss(output, fake_target)
loss.backward()

exit()

layer_norm_weights_file = open("data/tests/layer_norm/layer_norm_weights.txt", "wb")
layer_norm_bias_file = open("data/tests/layer_norm/layer_norm_bias.txt", "wb")
test_input_file = open("data/tests/layer_norm/test_input.txt", "wb")
expected_output_file = open("data/tests/layer_norm/expected_output.txt", "wb")
fake_target_file = open("data/tests/layer_norm/fake_target.txt", "wb")
loss_file = open("data/tests/layer_norm/expected_loss.txt", "wb")
layer_norm_weights_grad_file = open("data/tests/layer_norm/layer_norm_weights_grad.txt", "wb")
layer_norm_bias_grad_file = open("data/tests/layer_norm/layer_norm_bias_grad.txt", "wb")

write_fp32(layer_norm.weight, layer_norm_weights_file)
write_fp32(layer_norm.bias, layer_norm_bias_file)
write_fp32(test_input, test_input_file)
write_fp32(output, expected_output_file)
write_fp32(fake_target, fake_target_file)
write_fp32(torch.tensor(loss), loss_file)
write_fp32(layer_norm.weight.grad, layer_norm_weights_grad_file)
write_fp32(layer_norm.bias.grad, layer_norm_bias_grad_file)