"""Multi-view dataset loader. Supports X1..Xn+Y and X cell array formats."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import scipy.io
from config import DATA_DIR


class MultiViewDataset(Dataset):
    """Load multi-view .mat files from data/ or datasets/ directories.

    Supports two formats:
      - X1, X2, ..., Xn + Y  (flat arrays)
      - X cell array + gt/Y   (cell array, from datasets/)
    """

    def __init__(self, name):
        path = self._find_file(name)
        mat = scipy.io.loadmat(path)
        scaler = MinMaxScaler()

        # Format 1: X cell array + gt/Y (datasets/ directory)
        if 'X' in mat and ('gt' in mat or 'Y' in mat):
            label_key = 'gt' if 'gt' in mat else 'Y'
            self.labels = mat[label_key].astype(np.int32).flatten()
            X = mat['X']
            self.views, self.dims = [], []
            for i in range(X.shape[1]):
                v = X[0, i].astype(np.float32)
                if hasattr(v, 'toarray'):
                    v = v.toarray()
                if v.ndim > 2:
                    v = v.reshape(v.shape[0], -1)
                if v.shape[0] != len(self.labels):
                    v = v.T
                self.views.append(scaler.fit_transform(v))
                self.dims.append(v.shape[1])

        # Format 2: X1..Xn + Y (data/ directory)
        else:
            self.labels = mat['Y'].astype(np.int32).flatten()
            nv = len([k for k in mat.keys() if k.startswith('X') and k[1:].isdigit()])
            self.views, self.dims = [], []
            for i in range(nv):
                v = mat[f'X{i+1}'].astype(np.float32)
                if v.ndim > 2:
                    v = v.reshape(v.shape[0], -1)
                self.views.append(scaler.fit_transform(v))
                self.dims.append(v.shape[1])

        # Normalize labels to 0..C-1
        unique_labels = np.unique(self.labels)
        label_map = {ul: i for i, ul in enumerate(unique_labels)}
        self.labels = np.array([label_map[l] for l in self.labels])

        self.nv = len(self.views)
        self.nc = len(unique_labels)
        self.ns = self.views[0].shape[0]

    def _find_file(self, name):
        """Search for the .mat file in DATA_DIR (datasets folder)."""
        # DATA_DIR already points to the datasets folder
        p = os.path.join(DATA_DIR, f'{name}.mat')
        if os.path.exists(p):
            return p
        raise FileNotFoundError(f'{name}.mat not found in {DATA_DIR}')

    def __len__(self):
        return self.ns

    def __getitem__(self, i):
        return (
            [torch.from_numpy(v[i]) for v in self.views],
            torch.from_numpy(np.array(self.labels[i])),
            torch.from_numpy(np.array(i)).long(),
        )
