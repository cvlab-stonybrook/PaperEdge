import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from pathlib import Path

class DocUNet130(Dataset):
    def __init__(self, root_dir):
        self.data = []
        for k in range(1, 66):
            for m in range(1, 3):
                self.data.append(os.path.join(root_dir, '{}_{} copy.png'.format(k, m)))
                # self.data.append(os.path.join(root_dir, '{}_{}.png'.format(k, m)))

    def __len__(self):
        return 130

    def __getitem__(self, index):
        im = cv2.imread(self.data[index]).astype(np.float32) / 255.0
        im = im[:, :, (2, 1, 0)]
        im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        return im

    def readraw(self, index):
        # return the raw image in tensor format
        im = cv2.imread(self.data[index]).astype(np.float32) / 255.0
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        return im

class MiscData(Dataset):
    def __init__(self, is_raw=False):
        # self.is_raw = is_raw
        bsdir = '#path to the jpeg images you want to test'
        self.data = list(map(str, list(Path(bsdir).glob('*.jpg'))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        im = cv2.imread(self.data[index]).astype(np.float32) / 255.0
        im = im[:, :, (2, 1, 0)]
        im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        return im

    def readraw(self, index):
        # return the raw image in tensor format
        im = cv2.imread(self.data[index]).astype(np.float32) / 255.0
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        return im, self.data[index]