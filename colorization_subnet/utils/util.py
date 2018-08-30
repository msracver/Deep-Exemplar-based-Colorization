# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import cv2
import os
import numbers
import numpy as np
from skimage import color

import torch
import torchvision.utils as vutils
from torch.autograd import Variable

l_norm, ab_norm = 1., 1.
l_mean, ab_mean = 50., 0

def utf8_str(in_str):
    try:
        in_str = in_str.decode('UTF-8')
    except Exception:
        in_str = in_str.encode('UTF-8').decode('UTF-8')
    return in_str


def load_gray_image(img_path):
    img_path = utf8_str(img_path)
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img_gray


def load_rgb_image(img_path):
    img_path = utf8_str(img_path)
    img_rgb = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img_rgb


def resize_img_longside(img_rgb, long_size, interpolation):
    h, w = img_rgb.shape[:2]
    scalar = long_size / max(h, w)
    resized_img_rgb = cv2.resize(img_rgb, (int(w*scalar), int(h*scalar)), interpolation=interpolation)
    return resized_img_rgb


def resize_img_shortside(img_rgb, short_size, interpolation):
    h, w = img_rgb.shape[:2]
    scalar = short_size / min(h, w)
    resized_img_rgb = cv2.resize(img_rgb, (int(w*scalar), int(h*scalar)), interpolation=interpolation)
    return resized_img_rgb


def resize_img(img_rgb, dsize, interpolation):
    if isinstance(dsize, numbers.Number):
        dsize = (int(dsize), int(dsize))
    resized_img_rgb = cv2.resize(img_rgb, dsize, interpolation=interpolation)
    return resized_img_rgb


def rgb2lab_transpose(img_rgb):
    img_lab = color.rgb2lab(img_rgb).transpose((2, 0, 1))
    return img_lab


def center_lab_img(img_lab):

    img_lab_mc = img_lab / np.array((l_norm, ab_norm, ab_norm))[:, np.newaxis, np.newaxis] - np.array(
        (l_mean / l_norm, ab_mean / ab_norm, ab_mean / ab_norm))[:, np.newaxis, np.newaxis]
    return img_lab_mc


def center_l(l):
    l_mc = (l - l_mean) / l_norm
    return l_mc


def center_ab(ab):
    ab_mc = (ab - ab_mean) / ab_norm
    return ab_mc


def mult_mask(img_mask):
    mask_mult = 110
    return img_mask*mask_mult


def lab2rgb_transpose(img_l, img_ab):
    ''' INPUTS
            img_l     1xXxX     [0,100]
            img_ab     2xXxX     [-100,100]
        OUTPUTS
            returned value is XxXx3 '''
    pred_lab = np.concatenate((img_l, img_ab), axis=0).transpose((1, 2, 0))
    pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1)*255).astype('uint8')
    return pred_rgb

def lab2rgb(img_l, img_ab):
    ''' INPUTS
            img_l     XxXx1     [0,100]
            img_ab     XxXx2     [-100,100]
        OUTPUTS
            returned value is XxXx3 '''
    pred_lab = np.concatenate((img_l, img_ab), axis=2).astype('float64')
    pred_rgb = color.lab2rgb(pred_lab)
    pred_rgb = (np.clip(pred_rgb, 0, 1)*255).astype('uint8')
    return pred_rgb


def batch_lab2rgb_transpose_mc(img_l_mc, img_ab_mc):
    if isinstance(img_l_mc, Variable):
        img_l_mc = img_l_mc.data.cpu()
    if isinstance(img_ab_mc, Variable):
        img_ab_mc = img_ab_mc.data.cpu()

    if img_l_mc.is_cuda:
        img_l_mc = img_l_mc.cpu()
    if img_ab_mc.is_cuda:
        img_ab_mc = img_ab_mc.cpu()

    assert img_l_mc.dim()==4 and img_ab_mc.dim()==4, 'only for batch input'

    img_l = img_l_mc*l_norm + l_mean
    img_ab = img_ab_mc*ab_norm + ab_mean
    pred_lab = torch.cat((img_l, img_ab), dim=1)
    grid_lab = vutils.make_grid(pred_lab).numpy().astype('float64')
    grid_rgb = (np.clip(color.lab2rgb(grid_lab.transpose((1, 2, 0))), 0, 1)*255).astype('uint8')
    return grid_rgb


def lab2rgb_transpose_mc(img_l_mc, img_ab_mc):
    if isinstance(img_l_mc, Variable):
        img_l_mc = img_l_mc.data.cpu()
    if isinstance(img_ab_mc, Variable):
        img_ab_mc = img_ab_mc.data.cpu()

    if img_l_mc.is_cuda:
        img_l_mc = img_l_mc.cpu()
    if img_ab_mc.is_cuda:
        img_ab_mc = img_ab_mc.cpu()

    assert img_l_mc.dim()==3 and img_ab_mc.dim()==3, 'only for batch input'

    img_l = img_l_mc*l_norm + l_mean
    img_ab = img_ab_mc*ab_norm + ab_mean
    pred_lab = torch.cat((img_l, img_ab), dim=0)
    grid_lab = pred_lab.numpy().astype('float64')
    grid_rgb = (np.clip(color.lab2rgb(grid_lab.transpose((1, 2, 0))), 0, 1)*255).astype('uint8')
    return grid_rgb


def mkdir_if_not(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def to_np(x):
    return x.data.cpu().numpy()


