from trainer import fitNoMask
from data import BettiMinkovskiDataset
import pickle
from sklearn.metrics import r2_score
from ConvNet1D import ConvNet1D
import torch
import numpy as np
from torch.optim import Adam
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
import random
from sklearn.model_selection import train_test_split

m = "Curves"
batch_size = 32
epochs = 150
lr=1e-3
metric_fn = r2_score
import torch.nn as nn


criterion = nn.MSELoss()



SEED = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

features  = pickle.load(open("/Users/superbusiness/Desktop/TDA_Porous_Media/Data/natures_minkovsky+betti.pkl", "rb"))
targets = pickle.load(open("/Users/superbusiness/Desktop/TDA_Porous_Media/Data/y_natutes.pkl", "rb"))

labels = [
    'Volume', 'Area', 'Curvature', 'Euler', 'Betti-0', 'Betti-1', 'Betti-2'
]

features_subsets = {
        "Minkowski + Euler + Betti0": [0, 1, 2, 3, 4],
        "Minkowski + Euler + Betti1": [0, 1, 2, 3, 5],
        "Betti": [4, 5, 6],
        "Minkowski": [0, 1, 2],
        "Volume + Area + Euler + Betti01": [0, 1, 3, 4, 5],
        "Minkowski + Euler + Betti012": [0, 1, 2, 3, 4, 5, 6],
        "Volume + Area + Euler + Betti012": [0, 1, 3, 4, 5, 6],
        "Volume + Area + Euler + Betti1": [0, 1, 3, 5],
        "Volume + Area + Euler + Betti0": [0, 1, 3, 4],
        "Minkowski + Betti012": [0, 1, 2, 4, 5, 6],
        "Minkowski + Betti01": [0, 1, 2, 4, 5],
        "Minkowski + Euler + Betti2": [0, 1, 2, 3, 6],
}


for name in features_subsets:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNet1D(len(features_subsets[name]), 50)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X = torch.tensor(features[:, :, features_subsets[name]].transpose([0, 2, 1]), dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,
        random_state=42
    )

    train_dataset = BettiMinkovskiDataset(x_train, y_train)
    test_dataset = BettiMinkovskiDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"{name} features")
    fitNoMask(model, optimizer, criterion, metric_fn, epochs, train_loader, test_loader)
