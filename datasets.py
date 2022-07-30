from pathlib import Path
import torch as T
from torch.utils.data import Dataset
import numpy as np


class CRDataset(Dataset):
    def __init__(self, root: str, files_list=None):
        if files_list is None:
            self.files = list(Path(root).iterdir())
        else:
            self.files = [Path(root) / f for f in files_list]

    def __getitem__(self, index) -> T.tensor:
        path = str(self.files[index % len(self.files)])
        bundle_array = np.load(path)
        return T.from_numpy(bundle_array)

    def __len__(self) -> int:
        return len(self.files)
