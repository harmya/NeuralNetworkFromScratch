import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import re

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

context_length = 256
batch_size = 64
d_embed = 128
num_heads = 8
num_decoder_blocks = 6
max_iterations = 4000
learning_rate = 4e-4

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# print(text[:500])

characters = sorted(list(set(text)))
vocab_size = len(characters)
stringToInt = {ch : i for i,ch in enumerate(characters)}
intToString = {i : ch for i,ch in enumerate(characters)}

def encode_string(s):
    return [stringToInt[ch] for ch in s]

def decode_string(v):
    return ''.join([intToString[i] for i in v])


data = torch.tensor(encode_string(text), dtype=torch.long)
n = int(0.85*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length+1] for i in ix])
    return x, y


example_x, example_y = get_batch('train')
print(example_x.shape, example_y.shape)
print((example_x[0][:10], example_y[0][101]))
print(decode_string(example_x[0][:50].tolist()))
print(decode_string(example_y[0][:50].tolist()))

class InputEmbeddings(nn.Module):
    def __init__(self, d_embed : int, vocab_size : int):
        super().__init__()
        self.d_embed = d_embed
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_embed) # embedding[10] 
        
    def forward(self, x):
        # this takes in a (batch_size, context_length) tensor and returns a (batch_size, context_length, d_embed) tensor 
        # it maps each token to its embedding
        return self.embedding(x) * math.sqrt(self.d_embed)

class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, context_length):
        super().__init__()
        self.d_embed = d_embed
        self.context_length = context_length
        # initialize the positional encoding of size (context_length, d_embed)
        # for each position in context_length, we have a d_embed dimensional vector
        positional_encoding = torch.zeros(context_length, d_embed)
        # [0, 1, 2, 3, ..., context_length-1]
        position_index = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # 10000^(2i/d_embed) where i is the dimension of the positional encoding
        denominator = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
        
        positional_encoding[ : , 0::2] = torch.sin(position_index * denominator)
        positional_encoding[ : , 1::2] = torch.cos(position_index * denominator)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)
    
    def forward(self, x):
        return (x + self.positional_encoding[: , :x.shape[1], :])

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(d_embed, head_size, bias=False)
        self.query = nn.Linear(d_embed, head_size, bias=False)
        self.value = nn.Linear(d_embed, head_size, bias=False)
        self.mask = torch.tril(torch.ones(context_length, context_length))
        # mask is a lower triangular matrix of shape (context_length, context_length)
        # we use this to prevent the model from looking into the future
        # for each position i, we set the mask to 0 for all positions j where j > i
        # this way, the model can only attend to positions before i
    
    def forward(self, x):

        batch_size, sequence_length, feature_dimension = x.shape
        K = self.key(x)
        Q = self.query(x)
        q_kt = Q @ K.transpose(-2, -1) / np.sqrt(feature_dimension) 
        q_kt = q_kt.masked_fill(self.mask[:sequence_length, :sequence_length] == 0, float('-inf'))
        scaled_qkt = torch.nn.functional.softmax(q_kt, dim=-1)
        V = self.value(x)

        attention = scaled_qkt @ V
        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.linear_layer = nn.Linear(head_size * num_heads, d_embed) # head_size * num_heads = d_embed (usually)

    def forward(self, x):
        head_outputs = torch.cat([head(x) for head in self.heads], dim=-1) #[h1 h2 h3 ... hn]
        return self.linear_layer(head_outputs)

class FeedForward(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(d_embed, 4 * d_embed),
            nn.ReLU(),
            nn.Linear(4 * d_embed, d_embed)
        )
    
    def forward(self, x):
        return self.linear_layer(x)