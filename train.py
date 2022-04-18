import argparse
import json
import os
import random
import numpy as np
from pathlib import Path

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage

import torch
from torch import autograd
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# import pickle
from utils.handlers import VisPlot, CSVLogger
from networks.paperedge import GlobalWarper, LocalWarper, MaskLoss, WarperUtil, LocalLoss, SupervisedLoss
from data.doc3d import Doc3D, Doc3DDataAug
from data.diw import DIW, DIWDataAug

# parse the experiment configuration
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="experiment configuration files.")
exp_config = parser.parse_args().config
with open('configs/' + exp_config, 'r') as fid:
    args = json.load(fid)
print(args)
Path(args['exp_dir']).mkdir(parents=True, exist_ok=True)
device = 'cuda'

# netG is Enet and netL is Tnet in the paper.
netG = GlobalWarper().to(device)
netL = LocalWarper().to(device)

warpUtil = WarperUtil(64).to(device)
spvLoss = SupervisedLoss().to(device)
local_loss = LocalLoss()

if args['G_ckpt']:
    netG.load_state_dict(torch.load(args['G_ckpt'])['G'])
if args['L_ckpt']:
    netL.load_state_dict(torch.load(args['L_ckpt'])['L'])

optimizer_G = Adam(netG.parameters(), lr=args['lr_G'])
optimizer_L = Adam(netL.parameters(), lr=args['lr_L'])

# scheduler_G = StepLR(optimizer_G, 5, 0.1)
# scheduler_L = StepLR(optimizer_L, 5, 0.1)
scheduler_G = ReduceLROnPlateau(optimizer_G, factor=0.1, patience=2, verbose=True)
scheduler_L = ReduceLROnPlateau(optimizer_L, factor=0.1, patience=2, verbose=True)

mask_loss = MaskLoss(64)

class MixDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, ii):
        len_diw = len(self.datasets[1])
        jj = ii % len_diw
        # return self.datasets[1][jj]
        return self.datasets[0][ii], self.datasets[1][jj]


    def __len__(self):
        return len(self.datasets[0])

trn_loader = DataLoader(Doc3D(root_dir=args['doc3d_root']), batch_size=args['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
# trn_loader = DataLoader(MixDataset(Doc3D(root_dir=args['doc3d_root']), DIW(root_dir=args['diw'])), batch_size=args['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(Doc3D(root_dir=args['doc3d_root'], is_train=False), batch_size=args['batch_size'], num_workers=8, pin_memory=True)

doc3d_aug = Doc3DDataAug()
diw_aug = DIWDataAug()

model_id = 'demo'

# training step with a "_s" postfix is for supervised training with doc3d data.
# training step with a "_w" postfix is for weakly supervised training with both
# doc3d and diw data.
def train_G_step_s(engine, batch):
    netG.train()
    im, fm, bm, bg = batch
    im = im.to(device)
    fm = fm.to(device)
    bm = bm.to(device)
    bg = bg.to(device)
    with torch.no_grad():
        x, y = doc3d_aug(im, fm, bm, bg)
    netG.zero_grad()
    d = netG(x)
    loss = spvLoss.gloss(d, y)
    loss.backward()
    optimizer_G.step()
    dd = warpUtil.global_post_warp(d, 64)
    d = F.interpolate(dd, size=256, mode='bilinear', align_corners=True)
    fake_x = F.grid_sample(x, d.permute(0, 2, 3, 1), align_corners=True).detach()
    engine.state.img = [x, fake_x]
    return {'loss_G': loss.item()}

def validate_G_step_s(engine, batch):
    netG.eval()
    im, fm, bm, bg = batch
    im = im.to(device)
    fm = fm.to(device)
    bm = bm.to(device)
    bg = bg.to(device)
    with torch.no_grad():
        x, y = doc3d_aug(im, fm, bm, bg)
        d = netG(x)
        loss = spvLoss.gloss(d, y)
        engine.state.val_loss += loss.item()
        return {'loss_G': loss.item()}

def train_L_step_s(engine, batch):
    netL.train()
    im, fm, bm, bg = batch
    im = im.to(device)
    fm = fm.to(device)
    bm = bm.to(device)
    bg = bg.to(device)
    with torch.no_grad():
        x, y = doc3d_aug(im, fm, bm, bg)
    # pass the global warp net
        netG.eval()
        dg = netG(x)
        dg = warpUtil.global_post_warp(dg, 64)
        gs = F.interpolate(dg, 256, mode='bilinear', align_corners=True)
        xg = F.grid_sample(x, gs.permute(0, 2, 3, 1), align_corners=True)
    netL.zero_grad()
    xd = netL(xg)
    loss, _ = spvLoss.lloss(xd, y, dg)
    loss.backward()
    optimizer_L.step()
    fake_x = F.grid_sample(xg, F.interpolate(xd, 256, mode='bilinear', align_corners=True).permute(0, 2, 3, 1), align_corners=True)
    engine.state.img = [x, xg.detach(), fake_x.detach()]
    return {'loss_L': loss.item()}

def validate_L_step_s(engine, batch):
    im, fm, bm, bg = batch
    im = im.to(device)
    fm = fm.to(device)
    bm = bm.to(device)
    bg = bg.to(device)
    with torch.no_grad():
        x, y = doc3d_aug(im, fm, bm, bg)
        netG.eval()
        netL.eval()
        dg = netG(x)
        dg = warpUtil.global_post_warp(dg, 64)
        gs = F.interpolate(dg, 256, mode='bilinear', align_corners=True)
        xg = F.grid_sample(x, gs.permute(0, 2, 3, 1), align_corners=True)
        xd = netL(xg)
        loss, _ = spvLoss.lloss(xd, y, dg)
        engine.state.val_loss += loss.item()
        return {'loss_L': loss.item()}

def train_G_step_w(engine, batch):
    # doc3d data
    loss0 = train_G_step_s(engine, batch[0])['loss_G']
    # diw data
    x, xm, bg = batch[1]
    x = x.to(device)
    xm = xm.to(device)
    bg = bg.to(device)
    with torch.no_grad():
        x, xm = diw_aug(x, xm, bg)
    netG.zero_grad()
    d = netG(x)
    dd = F.interpolate(d, 64, mode='bilinear', align_corners=True)
    loss1, _ = mask_loss(dd, xm, 64)
    # weight
    loss1 *= 0.1
    loss1.backward()
    optimizer_G.step()
    dd = warpUtil.global_post_warp(d, 64)
    d = F.interpolate(dd, size=256, mode='bilinear', align_corners=True)
    fake_x = F.grid_sample(x, d.permute(0, 2, 3, 1), align_corners=True).detach()
    engine.state.img = [x, fake_x, xm]
    return {'loss0': loss0, 'loss1': loss1.item()}

def train_L_step_w(engine, batch):
    # doc3d data
    netL.train()
    im, fm, bm, bg = batch[0]
    im = im.to(device)
    fm = fm.to(device)
    bm = bm.to(device)
    bg = bg.to(device)
    with torch.no_grad():
        x, y = doc3d_aug(im, fm, bm, bg)
    # pass the global warp net
        netG.eval()
        dg = netG(x)
        dg = warpUtil.global_post_warp(dg, 64)
        gs = F.interpolate(dg, 256, mode='bilinear', align_corners=True)
        xg = F.grid_sample(x, gs.permute(0, 2, 3, 1), align_corners=True)
    netL.zero_grad()
    xd = netL(xg)
    loss0, xdh = spvLoss.lloss(xd, y, dg)
    loss0.backward()
    optimizer_L.step()

    # diw data
    x, xm, bg = batch[1]
    x = x.to(device)
    xm = xm.to(device)
    bg = bg.to(device)
    with torch.no_grad():
        x, xm = diw_aug(x, xm, bg)
        netG.eval()
        dg = netG(x)
        dg = warpUtil.global_post_warp(dg, 64)
        gs = F.interpolate(dg, 256, mode='bilinear', align_corners=True)
        x = F.grid_sample(x, gs.permute(0, 2, 3, 1), align_corners=True)
    # generate warp
    tgs, invtgs = warpUtil.perturb_warp(xdh)
    xp = F.grid_sample(x, F.interpolate(tgs, 256, mode='bilinear', align_corners=True).permute(0, 2, 3, 1), align_corners=True)
    netL.zero_grad()
    xd = netL(x)
    xpd = netL(xp.detach())
    loss1 = local_loss.warp_diff_loss(xd, xpd, tgs.detach(), invtgs.detach())
    loss1 *= 0.1
    loss1.backward()
    optimizer_L.step()
    fake_x = F.grid_sample(x, F.interpolate(xd, 256, mode='bilinear', align_corners=True).permute(0, 2, 3, 1), align_corners=True)
    fake_xp = F.grid_sample(xp, F.interpolate(xpd, 256, mode='bilinear', align_corners=True).permute(0, 2, 3, 1), align_corners=True)
    engine.state.img = [x.detach(), xp.detach(), fake_x.detach(), fake_xp.detach()]
    return {'loss0': loss0, 'loss1': loss1.item()}

trainer = Engine(train_G_step_s)
# trainer = Engine(train_L_step_s)
# trainer = Engine(train_G_step_w)
# trainer = Engine(train_L_step_w)
# trainer.state.metrics['loss_val'] = 0

validator = Engine(validate_G_step_s)
# validator = Engine(validate_L_step_s)
@trainer.on(Events.EPOCH_COMPLETED)
def validate(engine):
    validator.state.val_loss = 0
    validator.run(val_loader)
    val_loss = validator.state.val_loss / len(val_loader)
    trainer.state.metrics['loss_val'] = val_loss
    print(f'Validation loss: {val_loss}')

@trainer.on(Events.EPOCH_COMPLETED)
def update_lr(engine):
    scheduler_G.step(engine.state.metrics['loss_val'])
    # scheduler_L.step(engine.state.metrics['loss_val'])

RunningAverage(alpha=0.9, output_transform=lambda x: x['loss_G']).attach(trainer, 'loss_G')
# RunningAverage(alpha=0.9, output_transform=lambda x: x['loss_L']).attach(trainer, 'loss_L')
# RunningAverage(alpha=0.9, output_transform=lambda x: x['loss0']).attach(trainer, 'loss0')
# RunningAverage(alpha=0.9, output_transform=lambda x: x['loss1']).attach(trainer, 'loss1')



monitoring_metrics = ['loss_G']
# monitoring_metrics = ['loss_L']
# monitoring_metrics = ['loss0', 'loss1']

# model checkpoint
ckpt_hdl = ModelCheckpoint(args['exp_dir'], model_id, n_saved=1, require_empty=False, score_function=lambda x: -x.state.metrics['loss_val'])
# ckpt_hdl = ModelCheckpoint(args['exp_dir'], model_id, n_saved=3, require_empty=False)
# test on same images
vis_plot = VisPlot(port=args['vis_port'], env='gnet')
# attach progress bar
pbar = ProgressBar(persist=True)
pbar.attach(trainer, metric_names=monitoring_metrics)
# vpbar = ProgressBar()
# vpbar.attach(validator)
# log csv
csv_logger = CSVLogger(os.path.join(args['exp_dir'], 'log_' + model_id + '.csv'))

trainer.add_event_handler(Events.EPOCH_COMPLETED(every=5), ckpt_hdl, {'G': netG, 'L': netL})
# trainer.add_event_handler(Events.EPOCH_COMPLETED, tst_vis())

# vis
@trainer.on(Events.ITERATION_COMPLETED(every=args['vis_freq']))
def plot_input_output():
    for ii in range(len(trainer.state.img)):
        vis_plot.plot_imgs(trainer.state.img[ii][:min(16, args['batch_size'])].cpu().numpy(), win=f'img{ii}')
    # # plot mesh
    # tt = trainer.state.img[-1][:2].permute(0, 2, 3, 1).detach().cpu().numpy()
    # vis_plot.plot_meshes(tt, win='mesh1')
    # tt = trainer.state.img[-1][:2].permute(0, 2, 3, 1).detach().cpu().numpy()
    # vis_plot.plot_meshes(tt, win='mesh2')
    # fig, axs = plt.subplots(1, 8)
    # tt = trainer.state.img[-1][:8].permute(0, 2, 3, 1).detach().cpu().numpy()
    # plt.clf()
    # for ii in range(8):
    #     t = tt[ii]
    #     axs[ii].pcolormesh(t[..., 0], t[..., 1], np.zeros_like(t[..., 0]), edgecolors='r')
    #     axs[ii].invert_yaxis()
    #     axs[ii].axis('equal')
    # vis.matplot(plt, env='gnet', win='mpl')

trainer.add_event_handler(Events.ITERATION_COMPLETED(every=args['vis_freq']), vis_plot.plot_loss, monitoring_metrics)
trainer.add_event_handler(Events.ITERATION_COMPLETED(every=args['vis_freq']), csv_logger, monitoring_metrics)


trainer.run(trn_loader, max_epochs=args['epochs'])