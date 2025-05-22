import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error


class ConvNet1D(nn.Module):
    def __init__(self, in_channels, seq_len):
        super(ConvNet1D, self).__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        self.module1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.module2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(96, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(self.module1(x))
        x = self.pool(self.module2(x))

        x = x.view(-1, 96)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
