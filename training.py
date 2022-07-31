#%%
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch as T
from torch.utils.data import DataLoader

from datasets import CRDataset
from models import CRTransformer

# params
N_EPOCHS = 6
LR = 1
TRAIN_PROP = 0.8
BS_TRAIN = 28
BS_TEST = 13
losses_file = "CR_losses.csv"

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
train_dl = DataLoader(train_dataset, batch_size=BS_TRAIN, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=BS_TEST, shuffle=False)

# model and training stuff (loss, optim, sched)
model = CRTransformer(device=device).to(device)
criterion = T.nn.CrossEntropyLoss()
optimizer = T.optim.SGD(model.parameters(), lr=LR)

df_losses = pd.DataFrame(index=range(N_EPOCHS))
# training loop w/ val
for n in range(N_EPOCHS):
    model.train()
    loss_per_epoch = 0
    test_loss_per_epoch = 0
    for b_i, batch in enumerate(tqdm(train_dl)):
        src, tgt = batch
        src = src.to(device=device)
        tgt = tgt.to(device=device)

        if b_i == 0:
            out, out_indexes = model(src, tgt, src_masked=False, tgt_masked=False)
        else:
            out, out_indexes = model(
                src, prev_outputs, src_masked=False, tgt_masked=False
            )
        prev_outputs = out_indexes.clone().detach()

        tgt = tgt.to(dtype=T.long)  # for loss computation
        loss = criterion(out[:, -1, :], tgt[:, -1])
        loss.backward()
        T.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        loss_per_epoch += loss.item()
    loss_per_epoch /= len(train_dl)
    df_losses.at[n, "Training loss"] = loss_per_epoch

    model.eval()
    for b_i, batch in enumerate(test_dl):
        src, tgt = batch
        src = src.to(device=device)
        tgt = tgt.to(device=device)

        if b_i == 0:
            out, out_indexes = model(src, tgt, src_masked=False, tgt_masked=False)
        else:
            out, out_indexes = model(
                src, prev_outputs, src_masked=False, tgt_masked=False
            )
        prev_outputs = out_indexes.clone().detach()

        tgt = tgt.to(dtype=T.long)  # for loss computation
        loss = criterion(out[:, -1, :], tgt[:, -1])
        test_loss_per_epoch += loss.item()
    test_loss_per_epoch /= len(test_dl)

    df_losses.at[n, "Test loss"] = test_loss_per_epoch
    print(
        f"Epoch {n} --- Training Loss:{loss_per_epoch:.4f} --- Test Loss:{test_loss_per_epoch:.4f}"
    )

df_losses.to_csv("logs/" + losses_file, index_label="Epochs")
T.save(model.state_dict(), "last_model.pth")
