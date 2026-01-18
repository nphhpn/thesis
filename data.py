import os
import numpy as np
from torch.utils.data import Dataset as TorchDataset, DataLoader


class Dataset(TorchDataset):
    def __init__(self, directory):
        self.directory = directory
        self.chips = os.listdir(os.path.join(directory, "data"))

    def __len__(self):
        return len(self.chips)
    
    def __getitem__(self, idx):
        chip = self.chips[idx]
        data = np.load(os.path.join(self.directory, "data", chip))
        label = np.load(os.path.join(self.directory, "label", chip))
        return data, label[None, :, :]


class TrainDataset(Dataset):
    def __init__(self, directory, seed=42):
        super().__init__(directory)
        self.random = np.random.default_rng(seed)

    def __len__(self):
        return len(self.chips) * 4

    def __getitem__(self, idx):
        data, label = super().__getitem__(idx//4)
        C, H, W = data.shape
        x = self.random.integers(H-256)
        y = self.random.integers(W-256)
        data = data[:, x:x+256, y:y+256]
        label = label[:, x:x+256, y:y+256]
        if self.random.random() < 0.5:
            data = np.flip(data, axis=1)
            label = np.flip(label, axis=1)
        if self.random.random() < 0.5:
            data = np.flip(data, axis=2)
            label = np.flip(label, axis=2)
        return data.copy(), label.copy()
    

def get_dataloaders(data_root, batch_size=4):
    train_dataset = TrainDataset(f"{data_root}/train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size*4, shuffle=True)

    valid_dataset = Dataset(f"{data_root}/valid")
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    test_dataset = Dataset(f"{data_root}/test")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return [train_dataloader, valid_dataloader, test_dataloader]