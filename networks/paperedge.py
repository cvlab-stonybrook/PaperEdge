import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils import spectral_norm as SN
# from torchvision.models.densenet import _DenseBlock
from .tps_warp import TpsWarp, PspWarp
from functools import partial
# import plotly.graph_objects as go
import random
import numpy as np
import cv2

torch.autograd.set_detect_anomaly(True)
# torch.manual_seed(0)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.actv = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.actv(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.actv(out)

        return out

def _make_layer(block, inplanes, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=norm_layer))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, 
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=True)
        return x

class GlobalWarper(nn.Module):
    def __init__(self):
        super(GlobalWarper, self).__init__()
        modules = [
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ]

        # encoder
        planes = [64, 128, 256, 256, 512, 512]
        strides = [2, 2, 2, 2, 2]
        blocks = [1, 1, 1, 1, 1]
        for k in range(len(planes) - 1):
            modules.append(_make_layer(BasicBlock, planes[k], planes[k + 1], blocks[k], strides[k]))
        self.encoder = nn.Sequential(*modules)

        # decoder
        modules = []
        planes = [512, 512, 256, 128, 64]
        strides = [2, 2, 2, 2]
        # tsizes = [3, 5, 9, 17, 33]
        blocks = [1, 1, 1, 1]
        for k in range(len(planes) - 1):
            # modules += [nn.Sequential(Interpolate(size=tsizes[k], mode='bilinear'), 
            #             _make_layer(BasicBlock, planes[k], planes[k + 1], blocks[k], 1))]
            modules += [nn.Sequential(nn.Upsample(scale_factor=strides[k], mode='bilinear', align_corners=True), 
                        _make_layer(BasicBlock, planes[k], planes[k + 1], blocks[k], 1))]
        # self.decoder = nn.ModuleList(modules)
        self.decoder = nn.Sequential(*modules)

        self.to_warp = nn.Sequential(nn.Conv2d(64, 2, 1))
        self.to_warp[0].weight.data.fill_(0.0)
        self.to_warp[0].bias.data.fill_(0.0)
    
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256))
        self.coord = torch.stack((ix, iy), dim=0).unsqueeze(0).to('cuda')
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64))
        ### note we mulitply a 0.9 so the network is initialized closer to GT. This is different from localwarper net
        self.basegrid = torch.stack((ix * 0.9, iy * 0.9), dim=0).unsqueeze(0).to('cuda')

        # # box filter
        # ksize = 7
        # p = int((ksize - 1) / 2)
        # self.pad_replct = partial(F.pad, pad=(p, p, p, p), mode='replicate')
        # bw = torch.ones(1, 1, ksize, ksize, device='cuda') / ksize / ksize
        # self.box_filter = partial(F.conv2d, weight=bw)



    def forward(self, im):
        # print(self.to_warp[0].weight.data)
        # coordconv
        B = im.size(0)
        c = self.coord.expand(B, -1, -1, -1).detach()
        t = torch.cat((im, c), dim=1)

        t = self.encoder(t)
        t = self.decoder(t)
        t = self.to_warp(t)

        gs = t + self.basegrid

        return gs

class LocalWarper(nn.Module):
    def __init__(self):
        super().__init__()
        modules = [
            nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ]
        # encoder
        planes = [64, 128, 256, 256, 512, 512]
        strides = [2, 2, 2, 2, 2]
        blocks = [1, 1, 1, 1, 1]
        for k in range(len(planes) - 1):
            modules.append(_make_layer(BasicBlock, planes[k], planes[k + 1], blocks[k], strides[k]))
        self.encoder = nn.Sequential(*modules)

        # decoder
        modules = []
        planes = [512, 512, 256, 128, 64]
        strides = [2, 2, 2, 2]
        # tsizes = [3, 5, 9, 17, 33]
        blocks = [1, 1, 1, 1]
        for k in range(len(planes) - 1):
            modules += [nn.Sequential(nn.Upsample(scale_factor=strides[k], mode='bilinear', align_corners=True), 
                        _make_layer(BasicBlock, planes[k], planes[k + 1], blocks[k], 1))]
        self.decoder = nn.Sequential(*modules)

        self.to_warp = nn.Sequential(nn.Conv2d(64, 2, 1))
        self.to_warp[0].weight.data.fill_(0.0)
        self.to_warp[0].bias.data.fill_(0.0)

        iy, ix = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256))
        self.coord = torch.stack((ix, iy), dim=0).unsqueeze(0).to('cuda')
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64))
        self.basegrid = torch.stack((ix, iy), dim=0).unsqueeze(0).to('cuda')

        # box filter
        ksize = 5
        p = int((ksize - 1) / 2)
        self.pad_replct = partial(F.pad, pad=(p, p, p, p), mode='replicate')
        bw = torch.ones(1, 1, ksize, ksize, device='cuda') / ksize / ksize
        self.box_filter = partial(F.conv2d, weight=bw)

    def forward(self, im):
        c = self.coord.expand(im.size(0), -1, -1, -1).detach()
        t = torch.cat((im, c), dim=1)

        # encoder
        t = self.encoder(t)
        t = self.decoder(t)
        t = self.to_warp(t)

        # # filter
        # t = self.pad_replct(t)
        # tx = self.box_filter(t[:, 0 : 1, ...])
        # ty = self.box_filter(t[:, 1 : 2, ...])
        # t = torch.cat((tx, ty), dim=1)  

        # bd condition
        t[..., 1, 0, :] = 0
        t[..., 1, -1, :] = 0
        t[..., 0, :, 0] = 0
        t[..., 0, :, -1] = 0

        gs = t + self.basegrid
        return gs

def gs_to_bd(gs):
    # gs: B 2 H W
    t = torch.cat([gs[..., 0, :], gs[..., -1, :], gs[..., 1 : -1, 0], gs[..., 1 : -1, -1]], dim=2).permute(0, 2, 1)
    # t: B 2(W + H - 1) 2
    return t

class MaskLoss(nn.Module):
    def __init__(self, gsize):
        super().__init__()
        self.tpswarper = TpsWarp(gsize)
        self.pspwarper = PspWarp()
        # self.imsize = imsize
        self.msk = torch.ones(1, 1, gsize, gsize, device='cuda')
        self.cn = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float, device='cuda').unsqueeze(0)

    def forward(self, gs, y, s):
        # resize gs to s*s
        B, _, s0, _ = gs.size()
        tgs = F.interpolate(gs, s, mode='bilinear', align_corners=True)

        # use only the boundary points
        srcpts = gs_to_bd(tgs)
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        t = torch.stack((ix, iy), dim=0).unsqueeze(0).to('cuda').expand_as(tgs)
        dstpts = gs_to_bd(t)

        tgs_f = self.tpswarper(srcpts, dstpts.detach())
        ym = self.msk.expand_as(y)
        yh = F.grid_sample(ym, tgs_f.permute(0, 2, 3, 1), align_corners=True)
        loss_f = F.l1_loss(yh, y)

        # forward/backward consistency loss
        tgs_b = self.tpswarper(dstpts.detach(), srcpts)
        # tgs_b = F.interpolate(tgs, s0, mode='bilinear', align_corners=True)
        yy = F.grid_sample(y, tgs_b.permute(0, 2, 3, 1), align_corners=True)
        loss_b = F.l1_loss(yy, ym)
        
        return loss_f + loss_b, tgs_f

    def _dist(self, x):
        # adjacent point distance
        # B, 2, n
        x = torch.cat([x[..., 0 : 1].detach(), x[..., 1 : -1], x[..., -1 : ].detach()], dim=2)
        d = x[..., 1:] - x[..., :-1]
        return torch.norm(d, dim=1)

# class TVLoss(nn.Module):
#     def __init__(self):
#         super(TVLoss, self).__init__()

#     def forward(self, gs):
#         loss = self._dist(gs[..., 1:], gs[..., :-1]) + self._dist(gs[..., 1:, :], gs[..., :-1, :])
#         return loss
    
#     def _dist(self, x1, x0):
#         d = torch.norm(x1 - x0, dim=1, keepdim=True)
#         d = torch.abs(d - torch.mean(d, dim=(2, 3), keepdim=True)).mean()
#         return d

class WarperUtil(nn.Module):
    def __init__(self, imsize):
        super().__init__()
        self.tpswarper = TpsWarp(imsize)
        self.pspwarper = PspWarp()
        self.s = imsize
    
    def global_post_warp(self, gs, s):
        # B, _, s0, _ = gs.size()
        gs = F.interpolate(gs, s, mode='bilinear', align_corners=True)
        # gs = F.interpolate(gs, s0, mode='bilinear', align_corners=True)
        # extract info
        m1 = gs[..., 0, :]
        m2 = gs[..., -1, :]
        n1 = gs[..., 0]
        n2 = gs[..., -1]
        # for x
        m1x_interval_ratio = m1[:, 0, 1:] - m1[:, 0, :-1]
        m1x_interval_ratio /= m1x_interval_ratio.sum(dim=1, keepdim=True)
        m2x_interval_ratio = m2[:, 0, 1:] - m2[:, 0, :-1]
        m2x_interval_ratio /= m2x_interval_ratio.sum(dim=1, keepdim=True)
        # interpolate all x ratio
        t = torch.stack([m1x_interval_ratio, m2x_interval_ratio], dim=1).unsqueeze(1)
        mx_interval_ratio = F.interpolate(t, (s, m1x_interval_ratio.size(1)), mode='bilinear', align_corners=True)
        mx_interval = (n2[..., 0 : 1, :] - n1[..., 0 : 1, :]).unsqueeze(3) * mx_interval_ratio
        # cumsum to x
        dx = torch.cumsum(mx_interval, dim=3) + n1[..., 0 : 1, :].unsqueeze(3)
        dx = dx[..., 1 : -1, :-1]
        # for y
        n1y_interval_ratio = n1[:, 1, 1:] - n1[:, 1, :-1]
        n1y_interval_ratio /= n1y_interval_ratio.sum(dim=1, keepdim=True)
        n2y_interval_ratio = n2[:, 1, 1:] - n2[:, 1, :-1]
        n2y_interval_ratio /= n2y_interval_ratio.sum(dim=1, keepdim=True)
        # interpolate all x ratio
        t = torch.stack([n1y_interval_ratio, n2y_interval_ratio], dim=2).unsqueeze(1)
        ny_interval_ratio = F.interpolate(t, (n1y_interval_ratio.size(1), s), mode='bilinear', align_corners=True)
        ny_interval = (m2[..., 1 : 2, :] - m1[..., 1 : 2, :]).unsqueeze(2) * ny_interval_ratio
        # cumsum to y
        dy = torch.cumsum(ny_interval, dim=2) + m1[..., 1 : 2, :].unsqueeze(2)
        dy = dy[..., :-1, 1 : -1]
        ds = torch.cat((dx, dy), dim=1)
        gs[..., 1 : -1, 1 : -1] = ds
        return gs

    def perturb_warp(self, dd):
        B = dd.size(0)
        s = self.s
        # -0.2 to 0.2
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        t = torch.stack((ix, iy), dim=0).unsqueeze(0).to('cuda').expand(B, -1, -1, -1)

        tt = t.clone()

        nd = random.randint(0, 4)
        for ii in range(nd):
            # define deformation on bd
            pm = (torch.rand(B, 1) - 0.5) * 0.2
            ps = (torch.rand(B, 1) - 0.5) * 1.95
            pt = ps + pm
            pt = pt.clamp(-0.975, 0.975)
            # put it on one bd
            # [1, 1] or [-1, 1] or [-1, -1] etc
            a1 = (torch.rand(B, 2) > 0.5).float() * 2 -1
            # select one col for every row
            a2 = torch.rand(B, 1) > 0.5
            a2 = torch.cat([a2, a2.bitwise_not()], dim=1)
            a3 = a1.clone()
            a3[a2] = ps.view(-1)
            ps = a3.clone()
            a3[a2] = pt.view(-1)
            pt = a3.clone()
            # 2 N 4
            bds = torch.stack([
                t[0, :, 1 : -1, 0], t[0, :, 1 : -1, -1], t[0, :, 0, 1 : -1], t[0, :, -1, 1 : -1]
            ], dim=2)

            pbd = a2.bitwise_not().float() * a1
            # id of boundary p is on
            pbd = torch.abs(0.5 * pbd[:, 0] + 2.5 * pbd[:, 1] + 0.5).long()
            # ids of other boundaries
            pbd = torch.stack([pbd + 1, pbd + 2, pbd + 3], dim=1) % 4
            # print(pbd)
            pbd = bds[..., pbd].permute(2, 0, 1, 3).reshape(B, 2, -1)            

            srcpts = torch.stack([
                t[..., 0, 0], t[..., 0, -1], t[..., -1, 0], t[..., -1, -1],
                ps.to('cuda')
            ], dim=2)
            srcpts = torch.cat([pbd, srcpts], dim=2).permute(0, 2, 1)
            dstpts = torch.stack([
                t[..., 0, 0], t[..., 0, -1], t[..., -1, 0], t[..., -1, -1],
                pt.to('cuda')
            ], dim=2)
            dstpts = torch.cat([pbd, dstpts], dim=2).permute(0, 2, 1)
            # print(srcpts)
            # print(dstpts)
            tgs = self.tpswarper(srcpts, dstpts)
            tt = F.grid_sample(tt, tgs.permute(0, 2, 3, 1), align_corners=True)

        nd = random.randint(1, 5)
        for ii in range(nd):

            pm = (torch.rand(B, 2) - 0.5) * 0.2
            ps = (torch.rand(B, 2) - 0.5) * 1.95
            pt = ps + pm
            pt = pt.clamp(-0.975, 0.975)

            srcpts = torch.cat([
                t[..., -1, :], t[..., 0, :], t[..., 1 : -1, 0], t[..., 1 : -1, -1],
                ps.unsqueeze(2).to('cuda')
            ], dim=2).permute(0, 2, 1)
            dstpts = torch.cat([
                t[..., -1, :], t[..., 0, :], t[..., 1 : -1, 0], t[..., 1 : -1, -1],
                pt.unsqueeze(2).to('cuda')
            ], dim=2).permute(0, 2, 1)
            tgs = self.tpswarper(srcpts, dstpts)
            tt = F.grid_sample(tt, tgs.permute(0, 2, 3, 1), align_corners=True)
        tgs = tt

        # sample tgs to gen invtgs
        num_sample = 512
        # n = (H-2)*(W-2)
        n = s * s
        idx = torch.randperm(n)
        idx = idx[:num_sample]
        srcpts = tgs.reshape(-1, 2, n)[..., idx].permute(0, 2, 1)
        dstpts = t.reshape(-1, 2, n)[..., idx].permute(0, 2, 1)
        invtgs = self.tpswarper(srcpts, dstpts)
        return tgs, invtgs

    def equal_spacing_interpolate(self, gs, s):
        def equal_bd(x, s):
            # x is B 2 n
            v0 = x[..., :-1] # B 2 n-1
            v = x[..., 1:] - x[..., :-1]
            vn = v.norm(dim=1, keepdim=True)
            v = v / vn
            c = vn.sum(dim=2, keepdim=True) #B 1 1
            a = vn / c
            b = torch.cumsum(a, dim=2)
            b = torch.cat((torch.zeros(B, 1, 1, device='cuda'), b[..., :-1]), dim=2)
            
            t = torch.linspace(1e-5, 1 - 1e-5, s).view(1, s, 1).to('cuda')
            t = t - b # B s n-1
            # print(t)
            
            tt = torch.cat((t, -torch.ones(B, s, 1, device='cuda')), dim=2) # B s n
            tt = tt[..., 1:] * tt[..., :-1] # B s n-1
            tt = (tt < 0).float()
            d = torch.matmul(v0, tt.permute(0, 2, 1)) + torch.matmul(v, (tt * t).permute(0, 2, 1)) # B 2 s
            # print(d)
            return d

        gs = F.interpolate(gs, s, mode='bilinear', align_corners=True)
        B = gs.size(0)
        dst_cn = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float, device='cuda').expand(B, -1, -1)
        src_cn = torch.stack([gs[..., 0, 0], gs[..., 0, -1], gs[..., -1, -1], gs[..., -1, 0]], dim=2).permute(0, 2, 1)
        M = self.pspwarper.pspmat(src_cn, dst_cn).detach()
        invM = self.pspwarper.pspmat(dst_cn, src_cn).detach()
        pgs = self.pspwarper(gs.permute(0, 2, 3, 1).reshape(B, -1, 2), M).reshape(B, s, s, 2).permute(0, 3, 1, 2)
        t = [pgs[..., 0, :], pgs[..., -1, :], pgs[..., :, 0], pgs[..., :, -1]]
        d = []
        for x in t:
            d.append(equal_bd(x, s))
        pgs[..., 0, :] = d[0]
        pgs[..., -1, :] = d[1]
        pgs[..., :, 0] = d[2]
        pgs[..., :, -1] = d[3]
        gs = self.pspwarper(pgs.permute(0, 2, 3, 1).reshape(B, -1, 2), invM).reshape(B, s, s, 2).permute(0, 3, 1, 2)
        gs = self.global_post_warp(gs, s)
        return gs
        


class LocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def identity_loss(self, gs):
        s = gs.size(2)
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        t = torch.stack((ix, iy), dim=0).unsqueeze(0).to('cuda').expand_as(gs)
        loss = F.l1_loss(gs, t.detach())
        return loss

    def direct_loss(self, gs, invtgs):
        loss = F.l1_loss(gs, invtgs.detach())
        return loss

    def warp_diff_loss(self, xd, xpd, tgs, invtgs):
        loss_f = F.l1_loss(xd, F.grid_sample(tgs, xpd.permute(0, 2, 3, 1), align_corners=True).detach())
        loss_b = F.l1_loss(xpd, F.grid_sample(invtgs, xd.permute(0, 2, 3, 1), align_corners=True).detach())
        loss = loss_f + loss_b
        return loss


class SupervisedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        s = 64
        self.tpswarper = TpsWarp(s)

    def fm2bm(self, fm):
        # B 3 N N
        # fm in [0, 1]
        B, _, s, _ = fm.size()
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        t = torch.stack((ix, iy), dim=0).unsqueeze(0).to('cuda').expand(B, -1, -1, -1)
        srcpts = []
        dstpts = []
        for ii in range(B):
            # mask
            m = fm[ii, 2]
            # z s
            z = torch.nonzero(m, as_tuple=False)
            num_sample = 512
            n = z.size(0)
            # print(n)
            idx = torch.randperm(n)
            idx = idx[:num_sample]
            dstpts.append(t[ii, :, z[idx, 0], z[idx, 1]])
            srcpts.append(fm[ii, : 2, z[idx, 0], z[idx, 1]] * 2 - 1)
        srcpts = torch.stack(srcpts, dim=0).permute(0, 2, 1)
        dstpts = torch.stack(dstpts, dim=0).permute(0, 2, 1)
        # z = torch.nonzero(torch.abs(srcpts - 0) < 1e-5, as_tuple=False)
        # print(z.size(0))
        # print(dstpts.min())
        # print(dstpts.max())
        bm = self.tpswarper(srcpts, dstpts)
        # bm[bm > 1] = 1
        # bm[bm < -1] = -1
        return bm
    
    def gloss(self, x, y):
        xbd = gs_to_bd(x)
        # y = self.fm2bm(y)
        y = F.interpolate(y, 64, mode='bilinear', align_corners=True)
        
        ybd = gs_to_bd(y).detach()
        loss = F.l1_loss(xbd, ybd.detach())
        return loss

    def lloss(self, x, y, dg):
        # sample tgs to gen invtgs
        B, _, s, _ = dg.size()
        iy, ix = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s))
        t = torch.stack((ix, iy), dim=0).unsqueeze(0).to('cuda').expand(B, -1, -1, -1)
        num_sample = 512
        # n = (H-2)*(W-2)
        n = s * s
        idx = torch.randperm(n)
        idx = idx[:num_sample]
        # srcpts = gs_to_bd(tgs)
        # srcpts = torch.cat([srcpts, tgs[..., 1 : -1, 1 : -1].reshape(-1, 2, n)[..., idx].permute(0, 2, 1)], dim=1)
        srcpts = dg.reshape(-1, 2, n)[..., idx].permute(0, 2, 1)
        # dstpts = gs_to_bd(t)
        # dstpts = torch.cat([dstpts, t[..., 1 : -1, 1 : -1].reshape(-1, 2, n)[..., idx].permute(0, 2, 1)], dim=1)
        dstpts = t.reshape(-1, 2, n)[..., idx].permute(0, 2, 1)
        invdg = self.tpswarper(srcpts, dstpts)
        # compute dl = \phi(dg^-1, y)
        dl = F.grid_sample(invdg, y.permute(0, 2, 3, 1), align_corners=True)
        dl = F.interpolate(dl, 64, mode='bilinear', align_corners=True)
        loss = F.l1_loss(x, dl.detach())

        # y = F.interpolate(y, 64, mode='bilinear', align_corners=True)
        # loss = F.l1_loss(F.grid_sample(dg.detach(), x.permute(0, 2, 3, 1), align_corners=True), y)

        return loss, dl.detach()

