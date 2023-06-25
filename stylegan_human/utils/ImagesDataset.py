# Copyright (c) SenseTime Research. All rights reserved.


import os
from torch.utils.data import Dataset
from PIL import Image

from utils.data_utils import make_dataset


class ImagesDataset(Dataset):

    def __init__(self, source_root, source_transform=None):
        self.source_paths = sorted(make_dataset(source_root))
        self.source_transform = source_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        
        if self.source_transform:
            from_im = self.source_transform(from_im)

        return fname, from_im

