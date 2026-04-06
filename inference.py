import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Llama2
from tokenizer import LlamaTokenizer
from dataloader import LlamaDataLoader

from datasets import load_dataset

import os
import re
from pathlib import Path
import math

torch.manual_seed(69)
global_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def generate(last_token, p, temp):
    """
    Using the Top-p sampling approach
    """
    temp_scaled = last_token / temp
    probs = torch.softmax(temp_scaled, dim=-1)
    sorted_tensor, idx = torch.sort(probs, descending=True)
    cumulative = 0
    i = 0
    while cumulative < p:
        cumulative += sorted_tensor[0, i].item()
        i += 1
    candidates = sorted_tensor[:i] / cumulative
    index = torch.multinomial(candidates, num_samples=1, replacement=True)
    return idx[0, index.item()]

path = '/Users/bardia/Desktop/llama2/llama_tok.model'
tokenizer = LlamaTokenizer(model_path=path)

model: Llama2 = Llama2(
    vocab_size=32000,
    seq_len=512,
    d_model=512,
    num_heads=8,
    n_kv_heads=4,
    num_layers=8,
    is_causal=True
)

folder = Path('/Users/bardia/Desktop/llama2/checkpoints')
maxy = 0
if folder.is_dir():
    pt_files = [item for item in folder.iterdir() if item.suffix == '.pt']
    if pt_files:
        for item in pt_files:
            num = int(re.search(r'model_epoch_(\d+)\.pt', str(item)).group(1))
            maxy = max(maxy, num)
        newest = folder / f'model_epoch_{maxy}.pt'
        state_dicty = torch.load(newest)
        # torch.compile is so annoying with saving model params so have to revert before compiling again
        unwanted_prefix = '_orig_mod.' 
        for k,v in list(state_dicty.items()): 
            if k.startswith(unwanted_prefix): 
                state_dicty[k[len(unwanted_prefix):]] = state_dicty.pop(k)
        print('we good')
        model.load_state_dict(state_dicty)

#Enter whatever prompt here. This is what's gonna get used for inference
prompt = 'Once upon a time'

tokenized_prompt = tokenizer.encode(prompt)
prompt_len = len(tokenized_prompt)
prompt_tensor = (torch.tensor(tokenized_prompt)).unsqueeze(0).to(device=global_device)

model.eval()
model.reset_cache()
with torch.no_grad():
    #prefill step
    logits = model(prompt_tensor)
    last_token = logits[:, -1, :]
    sample = generate(last_token, p=0.9, temp=0.8)

    #decode step
    i = 0
    desired_len = 35
    prompt_tensor = torch.cat([prompt_tensor, sample.view(1, 1)], dim=-1)
    while i < desired_len:
        if sample == tokenizer.eos_id:
            break
        logits = model(sample.view(1,1).to(device=global_device))
        sample = generate(logits, p=0.9, temp=0.8)
        prompt_tensor = torch.cat([prompt_tensor, sample.view(1, 1)], dim=-1)
        i += 1

    print(f"PREDICTED SEQUENCE: {tokenizer.decode(prompt_tensor[0, :].tolist())}")
