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