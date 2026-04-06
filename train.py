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

#this method is where temp sampling is done
def generate(model, prompt, p, temp):
    """
    Using the Top-p sampling approach
    """
    logits = model(prompt)[:, -1, :]
    temp_scaled = logits / temp
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

# custom early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

path = '/Users/bardia/Desktop/llama2/llama_tok.model'
tokenizer = LlamaTokenizer(model_path=path)

train_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
valid_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

allowed_len = None #change this to an int when only wanting to pull some of a large dataset

#this is where we create the corpus texts for train and val batches
if not os.path.isfile("train_corpus.txt") or not os.path.isfile("valid_corpus.txt"):
    print("CREATING TRAINING CORPUS AND VALIDATION CORPUS FILES")
    with open("train_corpus.txt", "w", encoding="utf-8") as f:
            county = 0
            for line in train_dataset["text"]:
                if allowed_len and county == allowed_len:
                    break
                if line.strip():
                    f.write(line + "\n")
                county += 1

    with open("valid_corpus.txt", "w", encoding="utf-8") as f:
            county = 0
            for line in valid_dataset["text"]:
                if county == allowed_len:
                    break
                if line.strip():
                    f.write(line + "\n")
                county += 1

model: Llama2 = Llama2(
    vocab_size=32000,
    seq_len=512,
    d_model=512,
    num_heads=8,
    n_kv_heads=4,
    num_layers=8,
    is_causal=True
)

#checking if we already have some saved model checkpoints. Loading the most recent if so
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

# this is where we can see roughly mem needed for model and what is actually available
if True:
    mem_needed = ((sum(p.numel() for p in model.parameters()) * 16 ) / (1024**3))
    mem_limit = (torch.mps.recommended_max_memory() / (1024**3))
    assert mem_needed < mem_limit, f"Memory Needed: {mem_needed} vs Memory Available: {mem_limit}"
    print(f"Memory Needed: {mem_needed} vs Memory Available: {mem_limit}")

#uncomment the try except block to test out different model size configs to see if they're feasible before hitting OOM
"""
try:
    print('CREATING ONE-SHOT DATALOADER')
    example_loader = LlamaDataLoader(
          corpus_path='train_corpus.txt',
          tokenizer=tokenizer,
          seq_len=model.seq_len,
          batch_size=8,
          device='mps'
        )
    print('FINISHED CREATING ONE-SHOT DATALOADER')
    optimiz = optim.AdamW(model.parameters())
    x, y = example_loader.__next__()
    print('FORWARD PASS ON MODEL')
    logs = model.forward(x)
    print('FINISHED FORWARD PASS ON MODEL')
    V = logs.shape[-1]
        
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loss = loss_fn(logs.reshape(-1, V), y.reshape(-1))

    train_loss.backward()
    torch.mps.empty_cache()
    optimiz.zero_grad()

except RuntimeError as e:
    print(e)
    print('-----------------')
    print('OOM ERROR, CHECK DIMS AND PARAMS!')
    print('MODEL CONFIG THAT CAUSED PROBLEMS:')
    print(f"BATCH SIZE: {example_loader.batch_size}")
    print(f"VOCAB SIZE: {model.vocab_size}")
    print(f"SEQ LEN: {model.seq_len}")
    print(f"D_MODEL: {model.d_model}")
    print(f"NUM HEADS: {model.num_heads}")
    print(f"KV HEADS: {model.n_kv_heads}")
    print(f"NUM LAYERS: {model.layers}")
    torch.mps.empty_cache()
    optimiz.zero_grad()
 
"""

print('CREATING ACTUAL TRAIN LOADER')
train_dataloader = LlamaDataLoader(corpus_path='train_corpus.txt',
                                   tokenizer=tokenizer,
                                   seq_len=model.seq_len,
                                   batch_size=4,
                                   device='mps',
                                   )
print('FINISHED CREATING ACTUAL TRAIN LOADER')
print('CREATING ACTUAL VALIDATION LOADER')
valid_dataloader = LlamaDataLoader(corpus_path='valid_corpus.txt',
                                   tokenizer=tokenizer,
                                   seq_len=model.seq_len,
                                   batch_size=4,
                                   device='mps'
                                   )
print('FINISHED CREATING ACTUAL VALIDATION LOADER')
print('DataLoader size', train_dataloader.__len__())

max_lr = 3e-4
min_lr = max_lr * 0.10
num_epochs = 50
t_max = (train_dataloader.__len__() * num_epochs) - 2000
optimizer = optim.AdamW(params=model.parameters(),betas=(0.9, 0.95),eps=1e-5,weight_decay=0.1,lr=3e-4)
linear_lr = optim.lr_scheduler.LinearLR(optimizer=optimizer,start_factor=1e-8,end_factor=1.0,total_iters=2000)
cos_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=t_max,eta_min=min_lr)
lr_scheduler = optim.lr_scheduler.ChainedScheduler([linear_lr, cos_lr])
loss_func = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=5, min_delta=0.005)
accumulation_steps = 64 // train_dataloader.batch_size

stop = False

print(f'TAKING OFF FROM EPOCH: {maxy}')
model = torch.compile(model)
for num in range(maxy, num_epochs):

    if stop:
        break
    
    train_dataloader.reset()
    epoch_loss = 0.0
    batch_count = 0

    model.train()
    scaler = torch.amp.GradScaler("mps") #scaling to prevent underflow

    for index, (x, y) in enumerate(train_dataloader): 
            #this part helps setup mixed precision correctly and efficiently
            with torch.autocast("mps", dtype=torch.float16):
                logits = model(x)
                V = logits.shape[-1]

                loss = loss_func(logits.reshape(-1, V), y.reshape(-1))

            epoch_loss += loss.detach().item()
            batch_count += 1

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            
            if (index + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            if index > 0 and index % 1000 == 0:
                print(f'BATCH: {index}')

    avg_loss = epoch_loss / batch_count
    print(f"TRAIN LOSS EPOCH {num}: {avg_loss}")
    print(f"TRAIN PPL EPOCH {num}: {math.exp(avg_loss)}")
    print()

    #validation branch
    if num > 0 and num % 2 == 0:
        model.eval()
        with torch.no_grad():
            x_val, y_val = next(valid_dataloader)

            valid_logits = model(x_val)
            V = valid_logits.shape[-1]

            valid_loss = loss_func(valid_logits.reshape(-1, V), y_val.reshape(-1))
            print(f"VALID LOSS EPOCH {num}: {valid_loss.detach().item()}")
            print(f"VALID PPL EPOCH {num}: {math.exp(valid_loss.detach().item())}")
            print()

            if early_stopping.__call__(val_loss=valid_loss):
                stop = True

            # Temperature sampling part
            prompt = x_val[0][:model.seq_len // 2].unsqueeze(0)
            target = x_val[0][model.seq_len // 2:]
            
            start_len = prompt.size(dim=-1)
            while prompt.size(dim=-1) != (start_len + 35):
                model.reset_cache()
                sample = generate(model, prompt, p=0.9, temp=0.8)
                if sample == tokenizer.eos_id:
                    break
                prompt = torch.cat([prompt, sample.view(1, 1)], dim=-1)

            print(f"PREDICTED SEQUENCE: {tokenizer.decode(prompt[0, start_len:].tolist())}")
            print(f"TARGET SEQUENCE: {tokenizer.decode(target[:35].tolist())}")
            print()
            model.reset_cache()
    
    #saving statedict so we can pick up on training
    if num > 0 and num % 2 == 0:
        folder = Path('/Users/bardia/Desktop/llama2/checkpoints')
        if not folder.is_dir():
            os.makedirs(folder)
        torch.save(model.state_dict(), folder / f'model_epoch_{num}.pt')


         

