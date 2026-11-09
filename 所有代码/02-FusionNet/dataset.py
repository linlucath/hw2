from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset


class WVDataset(Dataset):
    """In-memory WV3 HDF5 data, normalized to the 11-bit radiometric scale."""
    def __init__(self, filename):
        with h5py.File(Path(filename), 'r') as f:
            self.data = {k: torch.from_numpy(f[k][:].astype('float32')) / 2047.0
                         for k in ('lms', 'pan', 'gt')}

    def __len__(self):
        return len(self.data['gt'])

    def __getitem__(self, index):
        return tuple(self.data[k][index] for k in ('lms', 'pan', 'gt'))
