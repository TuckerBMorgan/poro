import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import math
FLASH = 0
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
        self.y = None
        self.attn_weights = None
        self.filled = None
        self.mask = None
        
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
        
        if FLASH:
            # flashattention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
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
            self.attn_weights = query_key
            self.attn_weights.retain_grad()
            append_string_to_file("./python_checkfile.txt", "$AttnWeights")
            write_floats_to_file("./python_checkfile.txt", query_key.detach().numpy().flatten().tolist())
            premask = self.bias[:,:,:T,:T]
            append_string_to_file("./python_checkfile.txt", "$Premask")
            write_floats_to_file("./python_checkfile.txt", premask.detach().numpy().flatten().tolist())
            att = query_key.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            self.filled = att
            self.filled.retain_grad()
            append_string_to_file("./python_checkfile.txt", "$Filled")
            write_floats_to_file("./python_checkfile.txt", att.detach().numpy().flatten().tolist())
            att = F.softmax(att, dim=-1)
            append_string_to_file("./python_checkfile.txt", "$Softmax")
            write_floats_to_file("./python_checkfile.txt", att.detach().numpy().flatten().tolist())
            y = att @ v
            self.y = y
            self.y.retain_grad()
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

    def add_weights_to_dict(self, weights_dict, id = ""):
        if self.c_attn.bias is not None:
            weights_dict[id + "c_attn.bias"] = self.c_attn.bias
        weights_dict[id + "c_attn.weight"] = self.c_attn.weight        
        weights_dict[id + "c_proj.weight"] = self.c_proj.weight
        if self.c_proj.bias is not None:
            weights_dict[id + "c_proj.bias"] = self.c_proj.bias


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

config = type('', (), {})()
config.n_embd = 768
config.n_head = 12
config.block_size = 1024
causal_self_attention = CausalSelfAttention(config)
test_input = torch.randn(2, 8, 768)
fake_target = torch.randn(2, 8, 768)
output = causal_self_attention(test_input)



file_names = [
    "./data/tests/causal_self_attention/causal_self_attention_query_weights.txt",
    "./data/tests/causal_self_attention/causal_self_attention_query_bias.txt",
    "./data/tests/causal_self_attention/causal_self_attention_key_weights.txt", 
    "./data/tests/causal_self_attention/causal_self_attention_key_bias.txt",
    "./data/tests/causal_self_attention/causal_self_attention_value_weights.txt",
    "./data/tests/causal_self_attention/causal_self_attention_value_bias.txt",
    "./data/tests/causal_self_attention/causal_self_attention_c_proj_weights.txt", 
    "./data/tests/causal_self_attention/causal_self_attention_c_proj_bias.txt",
    "./data/tests/causal_self_attention/fake_output.txt",
    "./data/tests/causal_self_attention/loss.txt",
    "./data/tests/causal_self_attention/fake_input.txt",
    "./data/tests/causal_self_attention/expected_output.txt",
    "./data/tests/causal_self_attention/causal_self_attention_c_proj_weights_grad.txt",
    "./data/tests/causal_self_attention/casual_self_attention_c_proj_bias_grad.txt",
    "./data/tests/causal_self_attention/causal_self_attention_query_weights_grad.txt",
    "./data/tests/causal_self_attention/causal_self_attention_query_bias_grad.txt",
    "./data/tests/causal_self_attention/causal_self_attention_key_weights_grad.txt",
    "./data/tests/causal_self_attention/causal_self_attention_key_bias_grad.txt",
    "./data/tests/causal_self_attention/causal_self_attention_value_weights_grad.txt",
    "./data/tests/causal_self_attention/causal_self_attention_value_bias_grad.txt"
]

causal_self_attention.c_proj.weight.retain_grad()
#for file_name in file_names:
#    open(file_name, "wb").close()
    
loss_fn = torch.nn.MSELoss()
loss = loss_fn(output, fake_target)
loss.backward()
print(causal_self_attention.attn_weights.grad)
print(causal_self_attention.attn_weights.grad.shape)

exit()
write_fp32(causal_self_attention.q.weight, open(file_names[0], "wb"))
write_fp32(causal_self_attention.q.bias, open(file_names[1], "wb"))
write_fp32(causal_self_attention.k.weight, open(file_names[2], "wb"))
write_fp32(causal_self_attention.k.bias, open(file_names[3], "wb"))
write_fp32(causal_self_attention.v.weight, open(file_names[4], "wb"))
write_fp32(causal_self_attention.v.bias, open(file_names[5], "wb"))
write_fp32(causal_self_attention.c_proj.weight, open(file_names[6], "wb"))
write_fp32(causal_self_attention.c_proj.bias, open(file_names[7], "wb"))
write_fp32(fake_target, open(file_names[8], "wb"))
write_fp32(loss, open(file_names[9], "wb"))
write_fp32(test_input, open(file_names[10], "wb"))
write_fp32(output, open(file_names[11], "wb"))
write_fp32(causal_self_attention.c_proj.weight.grad, open(file_names[12], "wb"))
write_fp32(causal_self_attention.c_proj.bias.grad, open(file_names[13], "wb"))
write_fp32(causal_self_attention.q.weight.grad, open(file_names[14], "wb"))
write_fp32(causal_self_attention.q.bias.grad, open(file_names[15], "wb"))
write_fp32(causal_self_attention.k.weight.grad, open(file_names[16], "wb"))
write_fp32(causal_self_attention.k.bias.grad, open(file_names[17], "wb"))
write_fp32(causal_self_attention.v.weight.grad, open(file_names[18], "wb"))
write_fp32(causal_self_attention.v.bias.grad, open(file_names[19], "wb"))








