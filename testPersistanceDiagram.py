import sys
sys.path.append("./src/")

import random
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import r2_score

import pickle
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR


from data import PersistenceTransformDataset, collate_fn

from PersistanceTransformer import PersistentHomologyTransformer
from trainer import fit
from sklearn.model_selection import train_test_split


images = pickle.load(open("/Users/superbusiness/Desktop/TDA_Porous_Media/Data/diagrams_natures.pkl", "rb"))
y_data = pickle.load(open("/Users/superbusiness/Desktop/TDA_Porous_Media/Data/y_natutes.pkl", "rb"))

SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



diagrams_train, diagrams_test, y_train, y_test = train_test_split(
    images, y_data,
    test_size=0.1,
    random_state=42
)

log_train = np.log10(np.asarray(y_train, dtype=np.float32) + 1e-6)
mean = log_train.mean()
std = log_train.std()
test_log  = np.log10(np.asarray(y_test, dtype=np.float32) + 1e-6)
train_norm = (log_train - mean) / std
test_norm  = (test_log  - mean) / std



m = "PHTX"
d_model = 64
d_hidden = 192
num_heads = 8
num_layers = 4
dropout = 0.1
batch_size = 4
lr = 2.5e-4
epochs = 250


eps = 6.7

dataset_train = PersistenceTransformDataset([], train_norm, diagrams_train, eps=eps)
dataset_test = PersistenceTransformDataset([], test_norm, diagrams_test, eps=eps)

dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True, collate_fn=collate_fn)
dataloader_test = DataLoader(dataset_test, batch_size, collate_fn=collate_fn)

cuda = "cuda:0"
device = torch.device(cuda) if torch.cuda.is_available() else torch.device("cpu")

model = PersistentHomologyTransformer(None, 3, 1, d_model, d_hidden, num_heads, num_layers, dropout)
optimizer = Adam(model.parameters(), lr)
loss_fn = nn.MSELoss()
metric_fn = r2_score

print("Data:\t\t eps={}".format(eps))
print("Model:\t\t d_model={}, d_hidden={}, num_heads={}, num_layers={}, dropout={}".format(d_model, d_hidden, num_heads, num_layers, dropout))
print("Optimization:\t lr={}, batch size={}, seed={}".format(lr, batch_size, SEED))
_, history_model = fit(model, optimizer, loss_fn, metric_fn, epochs, dataloader_train, dataloader_test, desc=m)
