import glob
import os

import torch
from torch.utils.data import Dataset


class CacheDataset(Dataset):

    def __init__(self, cache_dir=None, in_memory=False):
        self.data = []
        self.in_memory = in_memory
        if not in_memory:
            self.cache_dir = cache_dir
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            else:
                self.data = [f for f in glob.glob(os.path.join(cache_dir, "**", "*.pt"), recursive=True)]

    def cleanup(self):
        if not self.in_memory:
            for item in self.data:
                os.unlink(item)
        self.data = []

    def has_cache(self):
        return len(self.data) > 0

    def add_record(self, data):
        if not self.in_memory:
            file_to_save = os.path.join(self.cache_dir, '%d.pt' % len(self.data))
            torch.save(data, file_to_save)
            self.data.append(file_to_save)
        else:
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.in_memory:
            file_cache = self.data[idx]
            return torch.load(file_cache)
        return self.data[idx]
