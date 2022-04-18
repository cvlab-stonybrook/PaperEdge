import visdom
import numpy as np
import csv
import torch
from datetime import datetime
import os
import cv2
import random
import matplotlib.pyplot as plt

class VisPlot(object):
    def __init__(self, port=10086, env='main'):
        self.vis = visdom.Visdom(port=port)
        self.env = env
        self.vis.close('loss', env=env)

    def plot_loss(self, engine, monitor_metrics, win='loss'):
        self.vis.line(X=np.array([engine.state.iteration]), 
        # NOTE because we use RunningAverage to log the loss, we can retrieve these numbers from state.metrics
            Y=np.array([[engine.state.metrics[x] for x in monitor_metrics]]),
                 env=self.env, win=win, update='append')
    
    def plot_imgs(self, imgs, win='img', imhistory=False):
        imgs = np.clip(imgs, 1e-5, 1 - 1e-5)
        self.vis.images(imgs, env=self.env, win=win, opts={'caption': win, 'store_history': imhistory})

    def plot_meshes(self, ms, win='ms'):
        plt.close()
        n = ms.shape[0]
        nr = (n - 1) // 8 + 1
        fig, axs = plt.subplots(1, 2)
        axs = axs.ravel()
        # fig.clf()

        c = np.arange(256) / 255.0
        c = c.reshape((16, 16))
        for ii in range(2):
            t = ms[ii]
            axs[ii].pcolormesh(t[..., 0], t[..., 1], c, cmap='YlGnBu', edgecolors='black')
            axs[ii].set_xlim(-1, 1)
            axs[ii].set_ylim(-1, 1)
            axs[ii].invert_yaxis()
            # axs[ii].axis('equal', 'box')
            axs[ii].set_aspect('equal', 'box')
        # fig, axs = plt.subplots(1, 2)
        # axs = axs.ravel()
        # t = ms[0]
        # axs[0].pcolormesh(t[..., 0], t[..., 1], np.zeros_like(t[..., 0]), edgecolors='r')
        # axs[0].invert_yaxis()
        # axs[0].axis('equal', 'box')
        fig.tight_layout()
        self.vis.matplot(fig, env=self.env, win=win)

class CSVLogger(object):
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, engine, monitor_metrics):
        with open(self.filename, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            date_time = datetime.now().strftime('%m/%d/%Y-%H:%M:%S')
            writer.writerow([date_time, engine.state.iteration] + [engine.state.metrics[x] for x in monitor_metrics])

# class SaveRes(object):
#     def __init__(self, resdir='./'):
#         self.yp = []
#         self.resdir = resdir

#     def update(self, engine):
#         self.yp.append(engine.state.output[0][1].cpu().numpy())
        
#     def save(self, epoch_id):
#         self.yp = np.concatenate(self.yp)
#         savemat(os.path.join(self.resdir, 't{}.mat'.format(epoch_id)), \
#         {'yp': self.yp})
#         self.yp = []
#         # self.yp = []
#         # self.yg = []


