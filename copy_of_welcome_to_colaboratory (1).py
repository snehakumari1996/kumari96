


import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
  def __init__(self, embed_size, heads):
    super(MultiHeadAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dim = embed_size // heads
    assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"
    self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
  def forward(self, values, keys, query, mask):
# Get number of training examples
    N = query.shape[0]
    value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
# Split the embedding into self.heads different pieces
    values = values.reshape(N, self.heads, value_len, self.head_dim)
    keys = keys.reshape(N,self.heads, key_len,  self.head_dim)
    queries = query.reshape(N,  self.heads,query_len, self.head_dim)
    values = self.values(values)
    keys = self.keys(keys)
    queries = self.queries(queries)
    energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
    if mask is not None:
      energy = energy.masked_fill(mask == 0, float("-1e20"))
    attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
    out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
    out = self.fc_out(out)
    return out
class TransformerBlock(nn.Module):
  def __init__(self, embed_size, heads):
    super(TransformerBlock, self).__init__()
    self.attention = MultiHeadAttention(embed_size, heads)
    self.feed_forward = nn.Sequential(
    nn.Linear(embed_size, 4 * embed_size),
    nn.ReLU(),
    nn.Linear(4 * embed_size, embed_size),
    )
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)
  def forward(self, value, key, query, mask):
    attention = self.attention(value, key, query, mask)
    # Add skip connection, run through normalization and finally feed forward network
    x = self.norm1(attention + query)
    forward = self.feed_forward(x)
    # Add skip connection, run through normalization
    x = self.norm2(forward + x)
    return x
class GPT2(nn.Module):
  def __init__(self, embed_size, heads, num_layers, vocab_size):
    super(GPT2, self).__init__()
    self.embed_size = embed_size
    self.embedding = nn.Embedding(vocab_size, embed_size)
    self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_size, heads) for _ in range(num_layers)])
  def forward(self, x, mask):
    x = self.embedding(x)
    for transformer in self.transformer_blocks:
      x = transformer(x, x, x, mask)
    return x
# Example usage
vocab_size = 10000 # Replace with actual vocabulary size
model = GPT2(embed_size=512, heads=8, num_layers=6, vocab_size=vocab_size)
# Create a sample input tensor
input_tensor = torch.randint(0, vocab_size, (1, 10))
# Create a sample mask (assuming no padding for simplicity)
N = input_tensor.shape[0]
query_len = input_tensor.shape[1]

# Create a sample mask (assuming no padding for simplicity)
mask = torch.ones(N, 1, query_len, query_len)

# Forward pass
output = model(input_tensor, mask)
print("Output shape:", output.shape)

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # Split the embedding into self.heads different pieces
        values = values.reshape(N, self.heads, value_len, self.head_dim)
        keys = keys.reshape(N, self.heads, key_len, self.head_dim)
        queries = query.reshape(N, self.heads, query_len, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        # Add skip connection, run through normalization and finally feed forward network
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        # Add skip connection, run through normalization
        x = self.norm2(forward + x)
        return x

class GPT2(nn.Module):
    def __init__(self, embed_size, heads, num_layers, vocab_size):
        super(GPT2, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_size, heads) for _ in range(num_layers)])

    def forward(self, x, mask):
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x, x, x, mask)
        return x

# Example usage
vocab_size = 10000  # Replace with the actual vocabulary size
model = GPT2(embed_size=512, heads=8, num_layers=6, vocab_size=vocab_size)

# Create a sample input tensor
input_tensor = torch.randint(0, vocab_size, (1, 10))

# Get number of training examples
N = input_tensor.shape[0]
query_len = input_tensor.shape[1]

# Create a sample mask (assuming no padding for simplicity)
mask = torch.ones(N, 1, query_len, query_len)

# Forward pass
output = model(input_tensor, mask)
print("Output shape:", output.shape)

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        # Add skip connection, run through normalization and finally feed forward network
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        # Add skip connection, run through normalization
        x = self.norm2(forward + x)
        return x

class GPT2(nn.Module):
    def __init__(self, embed_size, heads, num_layers, vocab_size):
        super(GPT2, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_size, heads) for _ in range(num_layers)])

    def forward(self, x, mask):
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x, x, x, mask)
        return x

# Example usage
vocab_size = 10000  # Replace with the actual vocabulary size
model = GPT2(embed_size=512, heads=8, num_layers=6, vocab_size=vocab_size)

# Create a sample input tensor
input_tensor = torch.randint(0, vocab_size, (1, 10))

# Get number of training examples
N = input_tensor.shape[0]
query_len = input_tensor.shape[1]

# Create a sample mask (assuming no padding for simplicity)
mask = torch.ones(N, 1, query_len, query_len)

# Forward pass
output = model(input_tensor, mask)
print("Output shape:", output.shape)

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)
        return x

class GPT2(nn.Module):
    def __init__(self, vocab_size, embed_size=768, heads=12, num_layers=12):
        super(GPT2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_size, heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask):
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        x = self.fc_out(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage
vocab_size = 10000  # Replace with the actual vocabulary size
model = GPT2(vocab_size=vocab_size)

# Create a sample input tensor
input_tensor = torch.randint(0, vocab_size, (1, 10))

# Get number of training examples
N = input_tensor.shape[0]
query_len = input_tensor.shape[1]

# Create a sample mask (assuming no padding for simplicity)
mask = torch.ones(N, 1, query_len, query_len)
total_params = count_parameters(model)
print(f"Total trainable parameters: {total_params}")
# Forward pass
output = model(input_tensor, mask)
print("Output shape:", output.shape)

print(f"Output shape: {output.shape}, Model Parameters: {sum(p.numel() for p in gpt_model.parameters()):,} parameters")

