import torch
import torch.nn as nn
import torch.nn.functional as F

class TpsWarp(nn.Module):
    def __init__(self, s):
        super(TpsWarp, self).__init__()
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        self.gs = torch.stack((ix, iy), dim=2).reshape((1, -1, 2)).to('cuda')
        self.sz = s


    def forward(self, src, dst):
        # src and dst are B.n.2
        B, n, _ = src.size()
        # B.n.1.2
        delta = src.unsqueeze(2)
        delta = delta - delta.permute(0, 2, 1, 3)
        # B.n.n
        K = delta.norm(dim=3)
        # Rsq = torch.sum(delta**2, dim=3)
        # Rsq += torch.eye(n, device='cuda')
        # Rsq[Rsq == 0] = 1.
        # K = 0.5 * Rsq * torch.log(Rsq)
        # c = -150
        # K = torch.exp(c * Rsq)
        # K = torch.abs(Rsq - 0.5) - 0.5
        # WARNING: TORCH.SQRT HAS NAN GRAD AT 0
        # K = torch.sqrt(Rsq)
        # print(K)
        # K[torch.isnan(K)] = 0.
        P = torch.cat((torch.ones((B, n, 1), device='cuda'), src), 2)
        L = torch.cat((K, P), 2)
        t = torch.cat((P.permute(0, 2, 1), torch.zeros((B, 3, 3), device='cuda')), 2)
        L = torch.cat((L, t), 1)
        # LInv = L.inverse()
        # # wv is B.n+3.2
        # wv = torch.matmul(LInv, torch.cat((dst, torch.zeros((B, 3, 2), device='cuda')), 1))
        # the above implementation has stability problem near the boundaries
        wv = torch.solve(torch.cat((dst, torch.zeros((B, 3, 2), device='cuda')), 1), L)[0]

        # get the grid sampler
        s = self.gs.size(1)
        gs = self.gs
        delta = gs.unsqueeze(2)   
        delta = delta - src.unsqueeze(1)
        K = delta.norm(dim=3)
        # Rsq = torch.sum(delta**2, dim=3)
        # K = torch.exp(c * Rsq)
        # Rsq[Rsq == 0] = 1.
        # K =  0.5 * Rsq * torch.log(Rsq)
        # K = torch.abs(Rsq - 0.5) - 0.5
        # K = torch.sqrt(Rsq)
        # K[torch.isnan(K)] = 0.
        gs = gs.expand(B, -1, -1)
        P = torch.cat((torch.ones((B, s, 1), device='cuda'), gs), 2)
        L = torch.cat((K, P), 2)
        gs = torch.matmul(L, wv)
        return gs.reshape(B, self.sz, self.sz, 2).permute(0, 3, 1, 2)

class PspWarp(nn.Module):
    def __init__(self):
        super().__init__()

    def pspmat(self, src, dst):
        # B, 4, 2
        B, _, _ = src.size()
        s = torch.cat([
            torch.cat([src, torch.ones((B, 4, 1), device='cuda'), torch.zeros((B, 4, 3), device='cuda'), 
                        -dst[..., 0 : 1] * src[..., 0 : 1], -dst[..., 0 : 1] * src[..., 1 : 2]], dim=2),
            torch.cat([torch.zeros((B, 4, 3), device='cuda'), src, torch.ones((B, 4, 1), device='cuda'), 
                        -dst[..., 1 : 2] * src[..., 0 : 1], -dst[..., 1 : 2] * src[..., 1 : 2]], dim=2)
        ], dim=1)
        t = torch.cat([dst[..., 0 : 1], dst[..., 1 : 2]], dim=1)
        # M = s.inverse() @ t
        M = torch.solve(t, s)[0]
        # M is B 8 1
        return M
    
    def forward(self, xy, M):
        # permute M to B 1 8
        M = M.permute(0, 2, 1)
        t = M[..., 6] * xy[..., 0] + M[..., 7] * xy[..., 1] + 1
        u = (M[..., 0] * xy[..., 0] + M[..., 1] * xy[..., 1] + M[..., 2]) / t
        v = (M[..., 3] * xy[..., 0] + M[..., 4] * xy[..., 1] + M[..., 5]) / t
        return torch.stack((u, v), dim=2)
        # for ii in range(4):
        #     xy = src[:, ii : ii + 1, :]
        #     uv = dst[:, ii : ii + 1, :]
        #     t0 = [xy, torch.ones((B, 1, 1), device='cuda'), torch.zeros((B, 1, 3), device='cuda'), -uv[..., 0] * xy[..., 0], -uv[..., 0] * xy[..., 1]]
        #     t0 = torch.cat(t0, dim=2)
        #     t1 = [torch.zeros((B, 1, 3), device='cuda'), xy, torch.ones((B, 1, 1), device='cuda'), -uv[..., 1] * xy[..., 0], -uv[..., 1] * xy[..., 1]]
        #     t1 = torch.cat(t1, dim=2)

class IdwWarp(nn.Module):
    # inverse distance weighting
    def __init__(self, s):
        super().__init__()
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        self.gs = torch.stack((ix, iy), dim=2).reshape((1, -1, 2)).to('cuda')
        self.s = s

    def forward(self, src, dst):
        # B n 2
        B, n, _ = src.size()
        # B.n.1.2
        delta = src.unsqueeze(2)
        delta = delta - self.gs.unsqueeze(0)
        # B.n.K
        p = 1
        Rsq = torch.sum(delta**2, dim=3)**p
        w = 1 / Rsq
        # turn inf to [0...1...0]
        t = torch.isinf(w)
        idx = t.any(dim=1).nonzero()
        w[idx[:, 0], :, idx[:, 1]] = t[idx[:, 0], :, idx[:, 1]].float()
        wwx = w * dst[..., 0 : 1]
        wwx = wwx.sum(dim=1) / w.sum(dim=1)
        wwy = w * dst[..., 1 : 2]
        wwy = wwy.sum(dim=1) / w.sum(dim=1)
        # print(wwy.size())
        gs = torch.stack((wwx, wwy), dim=2).reshape(B, self.s, self.s, 2).permute(0, 3, 1, 2)
        return gs


if __name__ == "__main__":
    import cv2
    import numpy as np
    from hdf5storage import loadmat
    from visdom import Visdom
    vis = Visdom(port=10086)

    # bm_path = '/nfs/bigdisk/sagnik/swat3d/bm/7/2_471_7-ec_Page_375-5LI0001.mat'
    # img_path = '/nfs/bigdisk/sagnik/swat3d/img/7/2_471_7-ec_Page_375-5LI0001.png'

    # bm = loadmat(bm_path)['bm']
    # bm = (bm - 224) / 224.
    # bm = cv2.resize(bm, (64, 64), cv2.INTER_LINEAR).astype(np.float32)

    # im = cv2.imread(img_path) / 255.
    # im = im[..., ::-1].copy()
    # im = cv2.resize(im, (256, 256), cv2.INTER_AREA).astype(np.float32)
    # im = torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0).to('cuda')

    # x = np.random.choice(np.arange(64), 50, False)
    # y = np.random.choice(np.arange(64), 50, False)

    # src = torch.tensor([[x, y]], device='cuda', dtype=torch.float32).permute(0, 2, 1)
    # src = (src - 32) / 32.
    # dst = torch.from_numpy(bm[y, x, :]).unsqueeze(0).to('cuda')

    # # print(src.size())
    # # print(dst.size())

    # tpswarp = TpsWarp(64)
    # import time
    # t = time.time()
    # for _ in range(100):
    #     gs = tpswarp(src, dst)
    # print(f'time:{time.time() - t}')
    # gs = gs.view(-1, 64, 64, 2)

    # print(gs.size())
    # bm2x2 = F.interpolate(gs.permute(0, 3, 1, 2), size=256, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)

    # rim = F.grid_sample(im, bm2x2, align_corners=True)
    # vis.images(rim, win='sk3')
    tpswarp = TpsWarp(16)
    import matplotlib.pyplot as plt
    cn = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1], [-0.5, -1], [0, -1], [0.5, -1]], dtype=torch.float, device='cuda').unsqueeze(0)
    pn = torch.tensor([[-1, -0.5], [1, -1], [1, 1], [-1, 0.5], [-0.5, -1], [0, -0.5], [0.5, -1]], device='cuda').unsqueeze(0)
    pspwarp = PspWarp()
    # # print(cn.dtype)
    M = pspwarp.pspmat(cn[..., 0 : 4, :], pn[..., 0 : 4, :])
    invM = pspwarp.pspmat(pn[..., 0 : 4, :], cn[..., 0 : 4, :])
    # iy, ix = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 8))
    # gs = torch.stack((ix, iy), dim=2).reshape((1, -1, 2)).to('cuda')
    # t = pspwarp(gs, M).reshape(8, 8, 2).detach().cpu().numpy()
    # print(M)

    t = tpswarp(cn, pn)
    from tsdeform import WarperUtil
    wu = WarperUtil(16)
    tgs = wu.global_post_warp(t, 16, invM, M)

    
    t = tgs.permute(0, 2, 3, 1)[0].detach().cpu().numpy()

    plt.clf()
    plt.pcolormesh(t[..., 0], t[..., 1], np.zeros_like(t[..., 0]), edgecolors='r')
    plt.gca().invert_yaxis()
    plt.gca().axis('equal')
    vis.matplot(plt, env='grid', win='mpl')

