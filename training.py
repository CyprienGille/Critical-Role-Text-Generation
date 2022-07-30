import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch as T
from torch.utils.data import DataLoader

from datasets import CRDataset
from models import CRTransformer

# params
N_EPOCHS = 30
LR = 1
TRAIN_PROP = 0.85
BS = 8

ready_dir = "data/ready/"

device = "cuda" if T.cuda.is_available() else "cpu"

# datasets and dataloaders
all_files = os.listdir(ready_dir)
n_files = len(all_files)
all_idx = np.arange(0, n_files)
train_idx = all_idx[: int(TRAIN_PROP * n_files)]
test_idx = all_idx[int(TRAIN_PROP * n_files) :]
train_files = list(map(all_files.__getitem__, train_idx))
test_files = list(map(all_files.__getitem__, test_idx))


train_dataset = CRDataset(root=ready_dir, files_list=train_files)
test_dataset = CRDataset(root=ready_dir, files_list=test_files)
train_dl = DataLoader(train_dataset, batch_size=BS, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=BS, shuffle=False)

# model and training stuff (loss, optim, sched)
model = CRTransformer(device=device).to(device)
criterion = T.nn.CrossEntropyLoss()
optimizer = T.optim.SGD(model.parameters(), lr=LR)

# training loop w/ val
for n in range(N_EPOCHS):
    model.train()
    loss_per_epoch = 0
    test_loss_per_epoch = 0
    for b_i, batch in enumerate(tqdm(train_dl)):
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)

        if b_i == 0:
            out, out_indexes = model(src, tgt, src_masked=False, tgt_masked=False)
        else:
            out, out_indexes = model(
                src, prev_outputs, src_masked=False, tgt_masked=False
            )
        prev_outputs = out_indexes.clone().detach()

        loss = criterion(out, tgt)
        loss.backward()
        T.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        loss_per_epoch += loss.item()
    loss_per_epoch /= len(train_dl)

    model.eval()
    for b_i, batch in enumerate(test_dl):
        src, tgt = batch

        if b_i == 0:
            _, out_indexes = model(src, tgt, src_masked=True, tgt_masked=True)
        else:
            _, out_indexes = model(src, prev_outputs, src_masked=True, tgt_masked=True)
        prev_outputs = out_indexes.clone().detach()

        loss = criterion(out_indexes, tgt)
        test_loss_per_epoch += loss.item()
    test_loss_per_epoch /= len(test_dl)

    print(
        f"Epoch {n} --- Training Loss:{loss_per_epoch:.4f} --- Test Loss:{test_loss_per_epoch:.4f}"
    )

# saving the best model and the logs ?
