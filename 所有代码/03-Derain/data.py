"""In-memory paired rain datasets."""
import h5py
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class TestDatasetMat(Dataset):
    def __init__(self, file_path, to_nchw=True):
        self.file_path = file_path
        self.data = None
        data = sio.loadmat(file_path)
        def tensor(key):
            x = torch.from_numpy(data[key])
            if to_nchw and x.ndim == 4 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2).contiguous()
            return x
        self.inputs, self.targets = tensor('img'), tensor('gt')
        self.length = len(self.inputs)

    def __getitem__(self, index):
        index %= self.length
        return {'img': self.inputs[index], 'gt': self.targets[index]}

    def __len__(self):
        return self.length


class TrainValDatasetH5(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        with h5py.File(self.file_path, 'r') as data:
            self.inputs = torch.from_numpy(data['img'][...])
            self.targets = torch.from_numpy(data['gt'][...])
        self.length = len(self.inputs)

    def __getitem__(self, index):
        index %= self.length
        return {'img': self.inputs[index], 'gt': self.targets[index]}

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = TrainValDatasetH5('./derain_small/Rain200L_train_chunks.h5')
    print(len(dataset), dataset[0]['img'].shape)
