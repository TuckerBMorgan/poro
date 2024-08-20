import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(DecoderOnlyTransformer, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super(AttentionHead, self).__init__()
        self.query = nn.Linear(embed_dim, head_dim)
        self.key = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)
        self.scale = head_dim ** -0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        print(v)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        return attn_output, attn_weights

torch.manual_seed(1133)

# Example usage:
embed_dim = 10
head_dim = 1
x = torch.arange(0, 10).view(1, 10).float()

attention_head = AttentionHead(embed_dim, head_dim)
output, weights = attention_head(x)
print(output)
import json
'''
json.dump(attention_head.query.weight.tolist(), open('query.json', 'w'))
json.dump(attention_head.key.weight.tolist(), open('key.json', 'w'))
json.dump(attention_head.value.weight.tolist(), open('value.json', 'w'))

json.dump(attention_head.query.bias.tolist(), open('query_bias.json', 'w'))
json.dump(attention_head.key.bias.tolist(), open('key_bias.json', 'w'))
json.dump(attention_head.value.bias.tolist(), open('value_bias.json', 'w'))
'''