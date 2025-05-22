from gudhi.representations import PersistenceImage


import numpy as np
import gudhi as gd

import torch
import torch.nn as nn
import torch.optim as optim



from torch.utils.data import DataLoader


from tqdm import tqdm
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import r2_score
from LpConv import make_model_with_lpconv

np.set_printoptions(precision=3, linewidth=120, edgeitems=45, threshold=100)
torch.set_printoptions(precision=3, linewidth=120, edgeitems=45, threshold=100)
from trainer import fitNoMask
from data import PersistanceImages3KDataset


images = pickle.load(open("//Users/superbusiness/Desktop/TDA_Porous_Media/Data/diagrams_natures.pkl", "rb"))
y_data = pickle.load(open("/Users/superbusiness/Desktop/TDA_Porous_Media/Data/y_natutes.pkl", "rb"))

pi = PersistenceImage(bandwidth=2.6, resolution=[32, 32])

images0 = [x[x[:, 2] == 0] for x in images]
images1 = [x[x[:, 2] == 1] for x in images]
images2 = [x[x[:, 2] == 2] for x in images]

pers_images0 = [pi.fit_transform([x]) for x in tqdm(images0)]
pers_images1 = [pi.fit_transform([x]) for x in tqdm(images1)]
pers_images2 = [pi.fit_transform([x]) for x in tqdm(images2)]

pers_images = np.stack([pers_images0, pers_images1, pers_images2], axis=1)

SEED = 6

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

images_train, images_test, y_train, y_test = train_test_split(
    pers_images, y_data,
    test_size=0.1,
    random_state=42
)

m = "PI3KLPConv"
img_size = 32
batch_size = 32
epochs = 600
input_channels = 3
base_filters = 16
num_layers = 4
log2p = 1
lr=1e-4
model = make_model_with_lpconv(input_channels=input_channels, base_filters=base_filters, num_layers=num_layers, img_size=img_size, log2p=log2p, lp_learnable=True)



dataset_train = PersistanceImages3KDataset(images_train, y_train)
dataset_test = PersistanceImages3KDataset(images_test, y_test)

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32)


optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()



metric_fn = r2_score # accuracy, roc auc, f1


print("Model:\t\t img_size={}, input_channels={}, base_filters={}, num_layers={}, log2p={}".format(
    img_size, input_channels, base_filters, num_layers, log2p
))
print("Optimization:\t lr={}, batch size={}, epochs={}".format(
    lr, batch_size, epochs
))
_, history_model = fitNoMask(model, optimizer, criterion, metric_fn, epochs, dataloader_train, dataloader_test, desc=m)
