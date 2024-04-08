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