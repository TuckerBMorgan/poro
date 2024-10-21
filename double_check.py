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


class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, input):
        print(torch.pow(input, 3.0))
        print((1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0)))))
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config["n_embd"], 4 * config["n_embd"])
        self.gelu    = NewGELU()
        self.c_proj  = nn.Linear(4 * config["n_embd"], config["n_embd"])
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


torch.manual_seed(42)
mlp_config = {}
mlp_config["n_embd"] = 768

mlp = MLP(mlp_config)

liner_1_weight_file = open("data/tests/mlp/linear_1_weights.txt", "wb")
linear_1_bias_file = open("data/tests/mlp/linear_1_bias.txt", "wb")
liner_2_weight_file = open("data/tests/mlp/linear_2_weights.txt", "wb")
linear_2_bias_file = open("data/tests/mlp/linear_2_bias.txt", "wb")
test_input_file = open("data/tests/mlp/test_input.txt", "wb")
output_file = open("data/tests/mlp/output.txt", "wb")
fake_target_file = open("data/tests/mlp/fake_target.txt", "wb")
expected_loss = open("data/tests/mlp/expected_loss.txt", "wb")

linear_1_weight_grad_file = open("data/tests/mlp/linear_1_weight_grad.txt", "wb")
linear_1_bias_grad_file = open("data/tests/mlp/linear_1_bias_grad.txt", "wb")
linear_2_weight_grad_file = open("data/tests/mlp/linear_2_weight_grad.txt", "wb")
linear_2_bias_grad_file = open("data/tests/mlp/linear_2_bias_grad.txt", "wb")


test_input = torch.randn(1, 768)
fake_target = torch.randn(1, 768)

output = mlp(test_input)

loss = F.mse_loss(output, fake_target)

loss.backward()

write_fp32(mlp.c_fc.weight, liner_1_weight_file)
write_fp32(mlp.c_proj.weight, liner_2_weight_file)
write_fp32(mlp.c_fc.bias, linear_1_bias_file)
write_fp32(mlp.c_proj.bias, linear_2_bias_file)
write_fp32(test_input, test_input_file)
write_fp32(output, output_file)
write_fp32(fake_target, fake_target_file)
write_fp32(torch.tensor([[loss]]), expected_loss)
write_fp32(mlp.c_fc.weight.grad, linear_1_weight_grad_file)
write_fp32(mlp.c_proj.weight.grad, linear_2_weight_grad_file)
write_fp32(mlp.c_fc.bias.grad, linear_1_bias_grad_file)
write_fp32(mlp.c_proj.bias.grad, linear_2_bias_grad_file)
