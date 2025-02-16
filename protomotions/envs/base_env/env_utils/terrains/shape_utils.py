# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import skimage
from scipy import ndimage
from skimage.draw import bezier_curve, circle_perimeter, disk, polygon


def draw_disk(img_size=80, max_r=10, iterations=3):
    shape = (img_size, img_size)
    img = np.zeros(shape, dtype=np.uint8)
    x, y = np.random.uniform(max_r, img_size - max_r, size=(2))
    radius = int(np.random.uniform(max_r))
    rr, cc = disk((x, y), radius, shape=shape)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    return img


def draw_circle(img_size=80, max_r=10, iterations=3):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    r, c = np.random.uniform(max_r, img_size - max_r, size=(2,)).astype(int)
    radius = int(np.random.uniform(max_r))
    rr, cc = circle_perimeter(r, c, radius)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    img = ndimage.binary_dilation(img, iterations=1).astype(int)
    return img


def draw_curve(img_size=80, max_sides=10, iterations=3):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    r0, c0, r1, c1, r2, c2 = np.random.uniform(0, img_size, size=(6,)).astype(int)
    w = np.random.random()
    rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, w)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    img = ndimage.binary_dilation(img, iterations=iterations).astype(int)
    return img


def draw_polygon(img_size=80, max_sides=10):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    num_coord = int(np.random.uniform(3, max_sides))
    r = np.random.uniform(0, img_size, size=(num_coord,)).astype(int)
    c = np.random.uniform(0, img_size, size=(num_coord,)).astype(int)
    rr, cc = polygon(r, c)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    return img


def draw_ellipse(img_size=80, max_size=10):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    r, c, rradius, cradius = (
        np.random.uniform(max_size, img_size - max_size),
        np.random.uniform(max_size, img_size - max_size),
        np.random.uniform(1, max_size),
        np.random.uniform(1, max_size),
    )
    rr, cc = skimage.draw.ellipse(r, c, rradius, cradius)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    return img
