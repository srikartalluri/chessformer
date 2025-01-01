import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, GPT2LMHeadModel

from torch.utils.data import DataLoader
import torch.optim as optim
from tokenizer import ChessTokenizer

import gc
import chess
import numpy as np
from model import ChessGPTModel
from chessdata import ChessDataset

cpu_device = torch.device("cpu")

# Check if MPS backend is available
if torch.backends.mps.is_available():
    print("Using MPS backend")
    gpu_device = torch.device("mps")
    
    print("MPS backend is available!")
else:
    print("Using Cuda backend")
    gpu_device = torch.device("cuda")

