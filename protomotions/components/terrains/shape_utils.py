# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import skimage
from scipy import ndimage
from skimage.draw import bezier_curve, circle_perimeter, disk, polygon


def draw_disk(img_size=80, max_r=10, iterations=3):
    shape = (img_size, img_size)
    img = np.zeros(shape, dtype=np.int16)
    x, y = np.random.uniform(max_r, img_size - max_r, size=(2))
    radius = int(np.random.uniform(max_r))
    rr, cc = disk((x, y), radius, shape=shape)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    return img


def draw_circle(img_size=80, max_r=10, iterations=3):
    img = np.zeros((img_size, img_size), dtype=np.int16)
    r, c = np.random.uniform(max_r, img_size - max_r, size=(2,)).astype(int)
    radius = int(np.random.uniform(max_r))
    rr, cc = circle_perimeter(r, c, radius)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    img = ndimage.binary_dilation(img, iterations=1).astype(np.int16)
    return img


def draw_curve(img_size=80, max_sides=10, iterations=3):
    img = np.zeros((img_size, img_size), dtype=np.int16)
    r0, c0, r1, c1, r2, c2 = np.random.uniform(0, img_size, size=(6,)).astype(int)
    w = np.random.random()
    rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, w)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    img = ndimage.binary_dilation(img, iterations=iterations).astype(np.int16)
    return img


def draw_polygon(img_size=80, max_sides=10):
    img = np.zeros((img_size, img_size), dtype=np.int16)
    num_coord = int(np.random.uniform(3, max_sides))
    r = np.random.uniform(0, img_size, size=(num_coord,)).astype(int)
    c = np.random.uniform(0, img_size, size=(num_coord,)).astype(int)
    rr, cc = polygon(r, c)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    return img


def draw_ellipse(img_size=80, max_size=10):
    img = np.zeros((img_size, img_size), dtype=np.int16)
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
