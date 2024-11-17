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

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch

        self.q = nn.Linear(config.n_embd, config.n_embd)
        self.k = nn.Linear(config.n_embd, config.n_embd)
        self.v = nn.Linear(config.n_embd, config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        

        append_string_to_file("./python_checkfile.txt", "$Query")
        write_floats_to_file("./python_checkfile.txt", q.detach().numpy().flatten().tolist())
        append_string_to_file("./python_checkfile.txt", "$Key")
        write_floats_to_file("./python_checkfile.txt", k.detach().numpy().flatten().tolist())
        append_string_to_file("./python_checkfile.txt", "$Value")
        write_floats_to_file("./python_checkfile.txt", v.detach().numpy().flatten().tolist())

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        append_string_to_file("./python_checkfile.txt", "$QueryT")
        write_floats_to_file("./python_checkfile.txt", q.detach().numpy().flatten().tolist())

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        append_string_to_file("./python_checkfile.txt", "$KeyT")
        write_floats_to_file("./python_checkfile.txt", k.detach().numpy().flatten().tolist())
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        append_string_to_file("./python_checkfile.txt", "$ValueT")
        write_floats_to_file("./python_checkfile.txt", v.detach().numpy().flatten().tolist())
        

        # manual implementation of attention
        # this materializes the large (T,T) matrix for all the queries and keys
        key_super_tranposed = k.transpose(-2, -1)
        append_string_to_file("./python_checkfile.txt", "$KeyST")
        write_floats_to_file("./python_checkfile.txt", key_super_tranposed.detach().numpy().flatten().tolist())
        query_key = q @ key_super_tranposed
        append_string_to_file("./python_checkfile.txt", "$QueryKey")
        write_floats_to_file("./python_checkfile.txt", query_key.detach().numpy().flatten().tolist())
        denom = (1.0 / math.sqrt(k.size(-1)))
        query_key = query_key * denom
        append_string_to_file("./python_checkfile.txt", "$AttnWeights")
        write_floats_to_file("./python_checkfile.txt", query_key.detach().numpy().flatten().tolist())
        premask = self.bias[:,:,:T,:T]
        print(premask.shape)
        append_string_to_file("./python_checkfile.txt", "$Premask")
        write_floats_to_file("./python_checkfile.txt", premask.detach().numpy().flatten().tolist())
        att = query_key.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        append_string_to_file("./python_checkfile.txt", "$Filled")
        write_floats_to_file("./python_checkfile.txt", att.detach().numpy().flatten().tolist())
        att = F.softmax(att, dim=-1)
        append_string_to_file("./python_checkfile.txt", "$Softmax")
        write_floats_to_file("./python_checkfile.txt", att.detach().numpy().flatten().tolist())
        y = att @ v
        append_string_to_file("./python_checkfile.txt", "$AttnOutput")
        write_floats_to_file("./python_checkfile.txt", y.detach().numpy().flatten().tolist())
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        append_string_to_file("./python_checkfile.txt", "$AttnOutputReshape")
        write_floats_to_file("./python_checkfile.txt", y.detach().numpy().flatten().tolist())
        y = self.c_proj(y)
        append_string_to_file("./python_checkfile.txt", "$CProj")
        write_floats_to_file("./python_checkfile.txt", y.detach().numpy().flatten().tolist())
        # output projection
        return y

torch.set_printoptions(precision=8)
torch.manual_seed(42)

class Config:
    n_embd = 768
    n_head = 12
    block_size = 1024

config = Config()
causal_self_attention = CausalSelfAttention(config)



fake_input = torch.randn(1, 1024, 768)
expected_output = causal_self_attention(fake_input)
fake_output = torch.randn(1, 1024, 768)
loss = F.mse_loss(expected_output, fake_output)
loss.backward()



files = []
files.append(open("./data/tests/causal_self_attention/causal_self_attention_query_weights.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_query_bias.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_key_weights.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_key_bias.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_value_weights.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_value_bias.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_c_proj_weights.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_c_proj_bias.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/fake_output.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/loss.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/fake_input.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/expected_output.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_c_proj_weights_grad.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/casual_self_attention_c_proj_bias_grad.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_query_weights_grad.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_query_bias_grad.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_key_weights_grad.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_key_bias_grad.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_value_weights_grad.txt", "wb"))
files.append(open("./data/tests/causal_self_attention/causal_self_attention_value_bias_grad.txt", "wb"))



write_fp32(causal_self_attention.q.weight, files[0])
write_fp32(causal_self_attention.q.bias, files[1])
write_fp32(causal_self_attention.k.weight, files[2])
write_fp32(causal_self_attention.k.bias, files[3])
write_fp32(causal_self_attention.v.weight, files[4])
write_fp32(causal_self_attention.v.bias, files[5])
write_fp32(causal_self_attention.c_proj.weight, files[6])
write_fp32(causal_self_attention.c_proj.bias, files[7])
write_fp32(fake_output, files[8])
write_fp32(torch.tensor([loss]), files[9])
write_fp32(fake_input, files[10])
write_fp32(expected_output, files[11])
write_fp32(causal_self_attention.c_proj.weight.grad, files[12])
write_fp32(causal_self_attention.c_proj.bias.grad, files[13])
write_fp32(causal_self_attention.q.weight.grad, files[14])
write_fp32(causal_self_attention.q.bias.grad, files[15])
write_fp32(causal_self_attention.k.weight.grad, files[16])
write_fp32(causal_self_attention.k.bias.grad, files[17])
write_fp32(causal_self_attention.v.weight.grad, files[18])
write_fp32(causal_self_attention.v.bias.grad, files[19])
