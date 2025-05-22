import torch
from torch.utils.data import Dataset
import numpy as np

class PersistenceTransformDataset(Dataset):
    def __init__(self, data, y, diagrams, dim=3 ,idx=None, eps=None, return_X=False):
        super().__init__()

        self.return_X = return_X
        if self.return_X:
            self.X = data
        self.y = torch.tensor(y, dtype=torch.float32)

        D = torch.ones([len(diagrams), max(map(len, diagrams))+1, dim]) * torch.inf


        for i, dgm in enumerate(diagrams):

            # eps
            if eps is not None:
                eps_idx = (dgm[:,1] - dgm[:,0]) >= eps
                dgm = dgm[eps_idx]

            # idx
            if idx is not None:
                dgm_idx = torch.isin(dgm[:,-1], idx)
                dgm = dgm[dgm_idx]
                if dim == 3:
                    D[i,:len(dgm)] = dgm
                else:
                    D[i,:len(dgm)] = dgm[:,:-1]
            else:
                if dim == 3:
                    D[i,:len(dgm)] = dgm
                else:
                    D[i,:len(dgm)] = dgm[:,:-1]


        max_len = torch.argmax(D[:,:,0], axis=1).max()
        D = D[:,:max_len+1]

        shapes_train_id = np.array([len(dgm) for dgm in D])
        print(shapes_train_id.min(), shapes_train_id.mean(), shapes_train_id.max())

        self.D = D

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx] if self.return_X else None, self.D[idx], self.y[idx]


def collate_fn(batch):

    n_batch = len(batch)
    d_lengths = [int(torch.argmax(D[:,0])) for X_, D, y_ in batch]


    Ds = torch.ones([n_batch, max(d_lengths), 3]) * 0.
    D_masks = torch.zeros([n_batch, max(d_lengths)]).bool()
    ys = torch.zeros(n_batch).float()

    for i, (X_, D, y) in enumerate(batch):
        Ds[i][:d_lengths[i]] = D[:d_lengths[i]]
        D_masks[i][d_lengths[i]:] = True
        ys[i] = y

    return Ds, D_masks, ys

def collate_fn_filltered(batch):

    n_batch = len(batch)
    d_lengths = [int(torch.argmax(D[:,0])) for X_, D, y_ in batch]


    Ds = torch.ones([n_batch, max(d_lengths), 5]) * 0.
    D_masks = torch.zeros([n_batch, max(d_lengths)]).bool()
    ys = torch.zeros(n_batch).float()

    for i, (X_, D, y) in enumerate(batch):
        Ds[i][:d_lengths[i]] = D[:d_lengths[i]]
        D_masks[i][d_lengths[i]:] = True
        ys[i] = y

    return Ds, D_masks, ys



class PersistanceImagesDataset():
    def __init__(self, images, y):
        self.X = images
        self.y = y
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]
        img = np.asarray(img, dtype=np.float32).reshape(32,32)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()
        y = np.float32(self.y[idx])
        return img, y


class PersistanceImages3KDataset():
    def __init__(self, images, y):
        self.X = images
        self.y = y
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]
        img = np.asarray(img, dtype=np.float32).reshape(3,32,32)
        img = torch.from_numpy(img).float()
        y = np.float32(self.y[idx])
        return img, y


class BettiMinkovskiDataset():
    def __init__(self, features, y):
        self.X = features
        self.y = y
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        fit = self.X[idx]
        y = np.float32(self.y[idx])
        return fit, y
