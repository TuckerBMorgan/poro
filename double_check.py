import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import math

'''
def simple_test():
    test_input_a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    test_input_a.requires_grad = True
    test_input_b = torch.tensor([5.0, 6.0, 7.0, 8.0])
    test_input_b.requires_grad = True
    test_input = test_input_a + test_input_b
    test_input.retain_grad()
    test_mean = test_input.mean()
    test_mean.retain_grad()
    fake_output = torch.tensor([5.0, 10.0])
    diff = test_mean - fake_output
    diff_mean = diff.mean()
    
    diff_mean.backward()
    print(test_input.grad)
    
simple_test()

exit()
'''
class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        Args:
            normalized_shape (int or tuple): Input shape for the last dimension to normalize.
            eps (float): Small value to prevent division by zero.
            elementwise_affine (bool): Whether to learn scale and bias parameters.
        """
        super(CustomLayerNorm, self).__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            # Learnable parameters for scale (gamma) and bias (beta)
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.normalized = None
        self.std = None
        self.mean = None
        self.input_mind_mean = None

    def forward(self, input):
        # Compute mean and variance along the last dimension
        mean = input.mean(dim=-1, keepdim=True)


        self.mean = mean
        self.mean.retain_grad()

        var = input.var(dim=-1, keepdim=True, unbiased=False)
        
        self.var = var
        self.var.retain_grad()
        
        input_mind_mean = input - mean

        self.input_mind_mean = input_mind_mean
        self.input_mind_mean.retain_grad()
        # Normalize the input
        normalized = input_mind_mean / torch.sqrt(var + self.eps)

        # Apply scale and shift (if elementwise_affine is True)
        if self.elementwise_affine:
            normalized = normalized * self.weight + self.bias
        self.normalized = normalized
        self.normalized.retain_grad()
        return normalized

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
layer_norm = CustomLayerNorm(768)
test_input_a = torch.randn(4, 64, 768)
test_input_b = torch.randn(4, 64, 768)
test_input = test_input_a + test_input_b
test_input.requires_grad = True


test_input.requires_grad = True
test_target = torch.randn(4, 64, 768)

fake_target = "data/tests/layer_norm/fake_target.txt"
write_fp32(test_target, open(fake_target, "wb"))

#print(ref_layer_norm(test_input))

#for index in x_shape:


output = layer_norm(test_input)

diff = output - test_target

loss = F.mse_loss(output, test_target)
#print(test_target)
loss.backward()

#print(output)
print(test_input.grad)
exit()
#print(loss)
'''
layer_norm_weights_path = "data/tests/layer_norm/layer_norm_weights.txt"
layer_norm_bias_path = "data/tests/layer_norm/layer_norm_bias.txt"
test_input_path = "data/tests/layer_norm/test_input.txt"
expected_output_path = "data/tests/layer_norm/expected_output.txt"

expected_loss = "data/tests/layer_norm/expected_loss.txt"
layer_norm_weights_grad = "data/tests/layer_norm/layer_norm_weights_grad.txt"
layer_norm_bias_grad = "data/tests/layer_norm/layer_norm_bias_grad.txt"

write_fp32(layer_norm.weight, open(layer_norm_weights_path, "wb"))
write_fp32(layer_norm.bias, open(layer_norm_bias_path, "wb"))
write_fp32(test_input, open(test_input_path, "wb"))
write_fp32(output, open(expected_output_path, "wb"))

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