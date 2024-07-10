import math
import torch
from torch import nn


single_head_attention_head = nn.MultiheadAttention(embed_dim=2, num_heads=1)
# I want to print out the weights of the attention head

# The weights are stored in the `in_proj_weight` attribute of the attention head
weights = single_head_attention_head.in_proj_weight
# I want to print out the key, query, and value weights separately
# first the key weights
key_weights = weights[:2]
print(key_weights)
# next the query weights
query_weights = weights[2:4]
print(query_weights)
# finally the value weights
value_weights = weights[4:]

print(value_weights)
