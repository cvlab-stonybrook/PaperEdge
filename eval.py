import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from visdom import Visdom
import matplotlib.pyplot as plt

vis = Visdom(port=10086)
import os
from pathlib import Path

# parse the experiment configuration
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="evaluation configuration files.")
exp_config = parser.parse_args().config
with open('configs/' + exp_config, 'r') as fid:
    args = json.load(fid)

from data.benchmark import DocUNet130
d130 = DocUNet130(root_dir=args['docunet_benchmark_img_dir'])
from torch.utils.data import DataLoader
dloader = DataLoader(d130, batch_size=64, num_workers=2, pin_memory=True)

from networks.paperedge import GlobalWarper, LocalWarper, WarperUtil

netG = GlobalWarper().to('cuda')
# path to the Enet checkpoint
netG.load_state_dict(torch.load(args['Enet_ckpt'])['G'])
# netG.load_state_dict(torch.load('G_w_checkpoint_13820.pt')['G'])
netG.eval()
netL = LocalWarper().to('cuda')
# path to the Tnet checkpoint
netL.load_state_dict(torch.load('Tnet_ckpt')['L'])
# netL.load_state_dict(torch.load('L_w_checkpoint_27640.pt')['L'])
netL.eval()

warpUtil = WarperUtil(64).to('cuda')

with torch.no_grad():
    res = []
    res0 = []
    inx = []
    gs = []
    ls = []
    for x in dloader:
        inx.append(x)
        x = x.to('cuda')
        # global
        d = netG(x)
        d = warpUtil.global_post_warp(d, 64)
        gs.append(d)
        d = F.interpolate(d, size=256, mode='bilinear', align_corners=True)
        y0 = F.grid_sample(x, d.permute(0, 2, 3, 1), align_corners=True)
        # y0 = x.clone()

        # change `nit` to a number > 1 to test iteratively applying Tnet to refine the unwarping result.
        # seems to improve the result slightly. Needs more investigation. Not reported in the paper.
        nit = 1
        y = y0.clone()
        for it in range(nit):
            d = netL(y)
            if it > 0:
                d = F.grid_sample(pd, d.permute(0, 2, 3, 1), padding_mode='reflection', align_corners=True)
            pd = d
            d = F.interpolate(d, size=256, mode='bilinear', align_corners=True)
            y = F.grid_sample(y0, d.permute(0, 2, 3, 1), align_corners=True)

        # d = netL(y)
        d = F.interpolate(d, size=256, mode='bilinear', align_corners=True)
        d = d.clamp(-1.0, 1.0)
        y = F.grid_sample(y0, d.permute(0, 2, 3, 1), align_corners=True)
        res.append(y)
        res0.append(y0)
        ls.append(d)
    res0 = torch.cat(res0, dim=0)
    res = torch.cat(res, dim=0)
    inx = torch.cat(inx, dim=0)
    gs = torch.cat(gs, dim=0)
    ls = torch.cat(ls, dim=0)
    vis.images(inx.clamp(1e-3, 1 - 1e-3).detach().cpu().numpy(), env='benchmark', win='inx', opts={'caption': 'inx'})
    vis.images(res.clamp(1e-3, 1 - 1e-3).detach().cpu().numpy(), env='benchmark', win='res', opts={'caption': 'res'})
    vis.images(res0.clamp(1e-3, 1 - 1e-3).detach().cpu().numpy(), env='benchmark', win='res0', opts={'caption': 'res0'})

# replace the following two lines with the benchmark image path and target result path
img_dir = args['docunet_benchmark_img_dir']
dst_dir = args['dst_dir']
Path(dst_dir).mkdir(parents=True, exist_ok=True)
ct = 0
for k in range(1, 66):
    print(k)
    for m in range(1, 3):
        im = cv2.imread(os.path.join(img_dir, f'{k}_{m} copy.png')).astype(np.float32) / 255.0
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        im = im.to('cuda').unsqueeze(0)
        d = gs[ct : ct + 1, ...]
        d = F.interpolate(d, (im.size(2), im.size(3)), mode='bilinear', align_corners=True)
        y = F.grid_sample(im, d.permute(0, 2, 3, 1), align_corners=True).detach()

        d = ls[ct : ct + 1, ...]
        d = F.interpolate(d, (im.size(2), im.size(3)), mode='bilinear', align_corners=True)
        y = F.grid_sample(y, d.permute(0, 2, 3, 1), align_corners=True).detach()
        y = y.squeeze().permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(f'{dst_dir}/{k}_{m}.png', y*255.)
        ct += 1