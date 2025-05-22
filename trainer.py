import torch
import numpy as np
from tqdm import tqdm

cuda = "cuda:0"
device = torch.device(cuda) if torch.cuda.is_available() else torch.device("cpu")

def fit(model, optimizer, loss_fn, metric_fn, n_epochs, dataloader_train, dataloader_test, desc=None):

    history = np.zeros((n_epochs, 4, 2)) # n_epochs, n_metrics, train/test

    model.to(device)

    pbar = tqdm(range(n_epochs), desc=desc, bar_format="{desc:<12.12}{percentage:3.0f}%|{bar:5}{r_bar}")
    for epoch_idx in pbar:

        # train
        model.train()
        all_preds, all_targets = [], []
        loss_batches = np.zeros((len(dataloader_train)))
        for i, (X, mask, Y) in enumerate(dataloader_train):
            X, mask, Y = X.to(device), mask.to(device), Y.to(device)
            Y_hat = model(X, mask)
            Y = Y.view(-1, 1)
            loss_batch = loss_fn(Y_hat, Y)
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

            all_preds.append(Y_hat.detach().cpu())
            all_targets.append(Y.cpu())

            with torch.no_grad():
                loss_batches[i] = loss_batch.detach() # CE

        all_preds_train = torch.cat(all_preds, dim=0).numpy()
        all_targets_train = torch.cat(all_targets, dim=0).numpy()
        train_r2 = metric_fn(all_targets_train, all_preds_train)
        history[epoch_idx,0,0] = np.mean(loss_batches)
        history[epoch_idx,1,0] = train_r2

        # test
        model.eval()

        loss_batches = np.zeros((len(dataloader_test)))
        all_preds, all_targets = [], []

        for i, (X, mask, Y) in enumerate(dataloader_test):
            X, mask, Y = X.to(device), mask.to(device), Y.to(device).float()
            Y = Y.view(-1, 1)
            Y_hat = model(X, mask)
            loss_batches[i] = loss_fn(Y_hat, Y).detach() # CE
            all_preds.append(Y_hat.detach().cpu())
            all_targets.append(Y.cpu())

        all_preds_test = torch.cat(all_preds, dim=0).numpy()
        all_targets_test = torch.cat(all_targets, dim=0).numpy()
        test_r2 = metric_fn(all_targets_test, all_preds_test)
        history[epoch_idx,0,1] = np.mean(loss_batches)
        history[epoch_idx,1,1] = test_r2

        pbar.set_postfix({
            "trainL": f"{history[epoch_idx,0,0]:.4f}",
            "bTrainL": f"{np.min(history[:epoch_idx+1,0,0]):.4f}",
            "testL": f"{history[epoch_idx,0,1]:.4f}",
            "bTestL": f"{np.min(history[:epoch_idx+1,0,1]):.4f}",
            "tL@bTL": f"{history[np.argmin(history[:epoch_idx+1,0,0]),0,1]:.4f}",
            "testR2": f"{history[epoch_idx,1,1]:.4f}",
            "bTestR2": f"{np.max(history[:epoch_idx+1,1,1]):.4f}",
            "R2@bTL": f"{history[np.argmin(history[:epoch_idx+1,0,0]),1,1]:.4f}",
        })

    return model, history


def fitNoMask(model, optimizer, loss_fn, metric_fn, n_epochs, dataloader_train, dataloader_test, desc=None):

    history = np.zeros((n_epochs, 4, 2)) # n_epochs, n_metrics, train/test
    model.to(device)

    pbar = tqdm(range(n_epochs), desc=desc, bar_format="{desc:<12.12}{percentage:3.0f}%|{bar:5}{r_bar}")
    for epoch_idx in pbar:

        # train
        model.train()
        all_preds, all_targets = [], []
        loss_batches = np.zeros((len(dataloader_train)))
        for i, (X, Y) in enumerate(dataloader_train):
            X, Y = X.to(device), Y.to(device).float()
            Y_hat = model(X)
            Y = Y.view(-1, 1)
            loss_batch = loss_fn(Y_hat, Y)
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

            all_preds.append(Y_hat.detach().cpu())
            all_targets.append(Y.cpu())

            with torch.no_grad():
                loss_batches[i] = loss_batch.detach()

        all_preds_train = torch.cat(all_preds, dim=0).numpy()
        all_targets_train = torch.cat(all_targets, dim=0).numpy()
        train_r2 = metric_fn(all_targets_train, all_preds_train)
        history[epoch_idx,0,0] = np.mean(loss_batches)
        history[epoch_idx,1,0] = train_r2

        # test
        model.eval()

        loss_batches = np.zeros((len(dataloader_test)))
        all_preds, all_targets = [], []

        for i, (X, Y) in enumerate(dataloader_test):
            X, Y = X.to(device), Y.to(device).float()
            Y = Y.view(-1, 1)
            Y_hat = model(X)
            loss_batches[i] = loss_fn(Y_hat, Y).detach()
            all_preds.append(Y_hat.detach().cpu())
            all_targets.append(Y.cpu())

        all_preds_test = torch.cat(all_preds, dim=0).numpy()
        all_targets_test = torch.cat(all_targets, dim=0).numpy()
        test_r2 = metric_fn(all_targets_test, all_preds_test)
        history[epoch_idx,0,1] = np.mean(loss_batches)
        history[epoch_idx,1,1] = test_r2
        pbar.set_postfix({
            "trainL": f"{history[epoch_idx,0,0]:.4f}",
            "bTrainL": f"{np.min(history[:epoch_idx+1,0,0]):.4f}",
            "testL": f"{history[epoch_idx,0,1]:.4f}",
            "bTestL": f"{np.min(history[:epoch_idx+1,0,1]):.4f}",
            "tL@bTL": f"{history[np.argmin(history[:epoch_idx+1,0,0]),0,1]:.4f}",
            "testR2": f"{history[epoch_idx,1,1]:.4f}",
            "bTestR2": f"{np.max(history[:epoch_idx+1,1,1]):.4f}",
            "R2@bTL": f"{history[np.argmin(history[:epoch_idx+1,0,0]),1,1]:.4f}",
        })

    return model, history
