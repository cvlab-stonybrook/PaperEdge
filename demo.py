# -*- encoding: utf-8 -*-
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from networks.paperedge import GlobalWarper, LocalWarper, WarperUtil

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def load_img(img_path):
    im = cv2.imread(img_path).astype(np.float32) / 255.0
    im = im[:, :, (2, 1, 0)]
    im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
    im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
    return im


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Enet_ckpt', type=str,
                        default='models/G_w_checkpoint_13820.pt')
    parser.add_argument('--Tnet_ckpt', type=str,
                        default='models/L_w_checkpoint_27640.pt')
    parser.add_argument('--img_path', type=str, default='images/3.jpg')
    parser.add_argument('--out_dir', type=str, default='output')
    args = parser.parse_args()

    img_path = args.img_path
    dst_dir = args.out_dir
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    netG = GlobalWarper().to('cuda')
    netG.load_state_dict(torch.load(args['Enet_ckpt'])['G'])
    netG.eval()

    netL = LocalWarper().to('cuda')
    netL.load_state_dict(torch.load(args['Tnet_ckpt'])['L'])
    netL.eval()

    warpUtil = WarperUtil(64).to('cuda')

    gs_d, ls_d = None, None
    with torch.no_grad():
        x = load_img(img_path)
        x = x.unsqueeze(0)
        x = x.to('cuda')
        d = netG(x)  # d_E the edged-based deformation field
        d = warpUtil.global_post_warp(d, 64)
        gs_d = copy.deepcopy(d)

        d = F.interpolate(d, size=256, mode='bilinear', align_corners=True)
        y0 = F.grid_sample(x, d.permute(0, 2, 3, 1), align_corners=True)
        ls_d = netL(y0)
        ls_d = F.interpolate(ls_d, size=256, mode='bilinear', align_corners=True)
        ls_d = ls_d.clamp(-1.0, 1.0)

    im = cv2.imread(img_path).astype(np.float32) / 255.0
    im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
    im = im.to('cuda').unsqueeze(0)

    gs_d = F.interpolate(gs_d, (im.size(2), im.size(3)), mode='bilinear', align_corners=True)
    gs_y = F.grid_sample(im, gs_d.permute(0, 2, 3, 1), align_corners=True).detach()

    tmp_y = gs_y.squeeze().permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(f'{dst_dir}/result_gs.png', tmp_y * 255.)
    # 精调： 由ls_d还原为原始图像
    ls_d = F.interpolate(ls_d, (im.size(2), im.size(3)), mode='bilinear', align_corners=True)
    ls_y = F.grid_sample(gs_y, ls_d.permute(0, 2, 3, 1), align_corners=True).detach()
    ls_y = ls_y.squeeze().permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(f'{dst_dir}/result_ls.png', ls_y * 255.)
