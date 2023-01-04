# set up logging
import logging
import pandas as pd
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
# make deterministic
from mingpt.utils import set_seed
set_seed(42)
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from rdkit import Chem
from torch.utils.data import Dataset
from mingpt.trainer import Trainer
import math
from mingpt.model import GPT
class CharDataset(Dataset):
        def __init__(self, data, content):
                chars = sorted(list(set(content)))
                data_size, vocab_size = len(data), len(chars)
                print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
                self.stoi = {ch: i for i, ch in enumerate(chars)}
                self.itos = {i: ch for i, ch in enumerate(chars)}
                self.block_size = block_size
                self.vocab_size = vocab_size
                self.data = data

        def __len__(self):
                return math.ceil(len(self.data) / (self.block_size + 1))

        def __getitem__(self, idx):
                smiles = self.data[idx]
                len_smiles = len(smiles)


                dix = [self.stoi[s] for s in smiles]
                x = torch.tensor(dix[:-1], dtype=torch.long)
                y = torch.tensor(dix[1:], dtype=torch.long)
                return x, y
def batch_end_callback(trainer):
    if trainer.iter_num % 1 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
smiles = pd.read_csv('moses.csv')['SMILES']
lens = [len(i) for i in smiles]
max_len = max(lens)
smiles = [ i + str('<')*(max_len - len(i)) for i in smiles]
content = ' '.join(smiles)
block_size = max_len
train_dataset = CharDataset(smiles, content)

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = train_dataset.vocab_size # openai's model vocabulary
model_config.block_size = train_dataset.block_size # openai's model block_size (i.e. input context lengt
#mconf = model_config(train_dataset.vocab_size, train_dataset.block_size,n_layer=8, n_head=8, n_embd=256)
model = GPT(model_config)
train_config = Trainer.get_default_config()
train_config.learning_rate = 6e-4 # many possible options, see the file
train_config.max_iters = 500
train_config.batch_size = 32
trainer = Trainer(train_config, model, train_dataset)
trainer.set_callback('on_batch_end', batch_end_callback)
trainer.run()
molecules = []

context = "C"
for i in range(100):
    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
    y = model.generate( x, block_size, temperature=0.9, do_sample=True, top_k=5)[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])
    completion = completion.replace('<', '')
    print(completion)
    mol = Chem.MolFromSmiles(completion)
    if mol:
        molecules.append(completion)

print(molecules)