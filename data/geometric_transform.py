import cv2
import numpy as np
import os

import random

class PerspectiveTransform(object):
    def __init__(self, scale=0.1, size=256):
        super().__init__()
        self.s = scale
        self.z = size
        self.p = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]).astype(np.float32)
        x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        self.pp = np.stack((x.ravel(), y.ravel()), axis=1).astype(np.float32)
    def __call__(self):
        q = np.abs(np.random.randn(4, 2) * self.s)
        # make sure displacement is within the area
        s = self.p * -2 + 1
        q = self.p + q * s
        m = cv2.getPerspectiveTransform(q.astype(np.float32), self.p)
        qq = cv2.perspectiveTransform(self.pp[None, :, :], m)
        return qq.reshape((self.z, self.z, 2)) * self.z
        

class GridDistort(object):
    def __init__(self, num_steps=5, distort_limit=0.3, size=256):
        super().__init__()
        self.distort_limit = distort_limit
        self.num_steps = num_steps
        self.z = size

    def __call__(self):
        xsteps = [1 + random.uniform(-self.distort_limit, self.distort_limit) for i in range(self.num_steps + 1)]
        ysteps = [1 + random.uniform(-self.distort_limit, self.distort_limit) for i in range(self.num_steps + 1)]
        x_step = self.z // self.num_steps
        xx = np.zeros(self.z, np.float32)
        prev = 0
        for idx in range(self.num_steps + 1):
            x = idx * x_step
            start = int(x)
            end = int(x) + x_step
            if end > self.z:
                end = self.z
                cur = self.z
            else:
                cur = prev + x_step * xsteps[idx]

            xx[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        y_step = self.z // self.num_steps
        yy = np.zeros(self.z, np.float32)
        prev = 0
        for idx in range(self.num_steps + 1):
            y = idx * y_step
            start = int(y)
            end = int(y) + y_step
            if end > self.z:
                end = self.z
                cur = self.z
            else:
                cur = prev + y_step * ysteps[idx]

            yy[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        map_x, map_y = np.meshgrid(xx, yy)
        return np.stack((map_x, map_y), axis=2).astype(np.float32)

class ElasticTransform(object):
    def __init__(self, alpha_scale=(5, 15), sigma=(2, 10), ssize=64, tsize=256):
        super().__init__()
        self.alpha_scale = alpha_scale
        self.sigma = sigma
        self.ssize = ssize
        self.tsize = tsize

    def __call__(self):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        alpha = sigma * random.uniform(self.alpha_scale[0], self.alpha_scale[1])
        dx = np.random.rand(self.ssize, self.ssize).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
        dx *= alpha

        dy = np.random.rand(self.ssize, self.ssize).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
        dy *= alpha

        x, y = np.meshgrid(np.linspace(0, self.tsize, self.ssize), np.linspace(0, self.tsize, self.ssize))

        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)

        mm = np.stack((map_x, map_y), axis=2).astype(np.float32)
        # print(f'{alpha} \t {sigma}')

        return cv2.resize(mm, (self.tsize, self.tsize), interpolation=cv2.INTER_LINEAR)

        