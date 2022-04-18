import cv2
import numpy as np
import scipy.interpolate
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random
import time

from hdf5storage import loadmat

import kornia.augmentation as KA
import kornia.geometry.transform as KG

class Doc3D(Dataset):
    def __init__(self, root_dir, is_train=True, num=0):
        super(Doc3D, self).__init__()
        # self.is_train = is_train
        self.num = num
        # load the list of doc3d images
        if is_train:
            with open('./data/doc3d_trn.txt', 'r') as fid:
                self.X = fid.read().splitlines()
        else:
            with open('./data/doc3d_val.txt', 'r') as fid:
                self.X = fid.read().splitlines()
        self.X = [root_dir + '/img/' + t + '.png' for t in self.X]
        
        # load the background images
        with open('./data/bgtex.txt', 'r') as fid:
            self.bgtex = fid.read().splitlines()        

    def __len__(self):
        if self.num:
            return self.num
        else:
            return len(self.X)

    def __getitem__(self, index):
        # index = index % 10
        t = self.X[index]
        # print(t)
        im = cv2.imread(t).astype(np.float32) / 255.0
        im = im[..., ::-1]

        t = t.replace('/img/', '/wc/')
        t = t[:-3] + 'exr'
        wc = cv2.imread(t, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED).astype(np.float32)

        t = t.replace('/wc/', '/bm/')
        t = t[:-3] + 'mat'
        bm = loadmat(t)['bm']

        # random sample a background image
        ind = random.randint(0, len(self.bgtex) - 1)
        bg = cv2.imread(self.bgtex[ind]).astype(np.float32) / 255.0
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))

        im = torch.from_numpy(im.transpose((2, 0, 1)).copy())
        wc = torch.from_numpy(wc.transpose((2, 0, 1)).copy())
        bm = torch.from_numpy(bm.transpose((2, 0, 1)).copy())
        bg = torch.from_numpy(bg.transpose((2, 0, 1)).copy())

        return im, wc, bm, bg



class Doc3DDataAug(nn.Module):
    def __init__(self):
        super(Doc3DDataAug, self).__init__()
        self.cj = KA.ColorJitter(0.1, 0.1, 0.1, 0.1)
    
    def forward(self, img, wc, bm, bg):
        # tight crop
        mask = (wc[:, 0] != 0) & (wc[:, 1] != 0) & (wc[:, 2] != 0)
        
        B = img.size(0)
        c = torch.randint(20, (B, 5))
        img_list = []
        bm_list = []
        for ii in range(B):
            x_img = img[ii]
            x_bm = bm[ii]
            x_msk = mask[ii]
            y, x = x_msk.nonzero(as_tuple=True)
            minx = x.min()
            maxx = x.max()
            miny = y.min()
            maxy = y.max()
            x_img = x_img[:, miny : maxy + 1, minx : maxx + 1]
            x_msk = x_msk[None, miny : maxy + 1, minx : maxx + 1]

            # padding
            x_img = F.pad(x_img, c[ii, : 4].tolist())
            x_msk = F.pad(x_msk, c[ii, : 4].tolist())

            x_bm[0, :, :] = (x_bm[0, :, :] - minx + c[ii][0]) / x_img.size(2) * 2 - 1
            x_bm[1, :, :] = (x_bm[1, :, :] - miny + c[ii][2]) / x_img.size(1) * 2 - 1

            # replace bg
            if c[ii][-1] > 2:
                x_bg = bg[ii][:, :x_img.size(1), :x_img.size(2)]
            else:
                x_bg = torch.ones_like(x_img) * torch.rand((3, 1, 1), device=x_img.device)
            x_msk = x_msk.float()
            x_img = x_img * x_msk + x_bg * (1. - x_msk)

            # resize
            x_img = KG.resize(x_img[None, :], (256, 256))
            img_list.append(x_img)
            bm_list.append(x_bm)
        img = torch.cat(img_list)
        bm = torch.stack(bm_list)
        # jitter color
        img = self.cj(img)
        return img, bm


if __name__ == '__main__':
    dt = Doc3D()
    from visdom import Visdom
    vis = Visdom(port=10086)
    x, xt, y, yt, t = dt[999]

    vis.image(x.clamp(0, 1), opts={'caption': 'x'}, win='x')
    vis.image(xt.clamp(0, 1), opts={'caption': 'xt'}, win='xt')
    vis.image(y.clamp(0, 1), opts={'caption': 'y'}, win='y')
    vis.image(yt.clamp(0, 1), opts={'caption': 'yt'}, win='yt')
    vis.image(t, opts={'caption': 't'}, win='t')
