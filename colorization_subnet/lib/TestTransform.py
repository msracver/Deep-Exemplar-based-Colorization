# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

from skimage import color

import lib.functional as F


__all__ = ["Compose", "Concatenate", "ToTensor", "Normalize", "Resize", "CenterCrop",
           "RandomCrop", "RandomHorizontalFlip", "RGB2Lab"]

def CustomFunc(inputs, func, *args, **kwargs):
    im_l     = func(inputs[0], *args, **kwargs)
    im_ab    = func(inputs[1], *args, **kwargs)
    warp_ba  = func(inputs[2], *args, **kwargs)
    warp_aba = func(inputs[3], *args, **kwargs)
    # im_gbl_ab  = func(inputs[4], *args, **kwargs)
    # bgr_mc_im = func(inputs[5], *args, **kwargs)
    layer_data = [im_l, im_ab, warp_ba, warp_aba]
    # layer_data = [im_l, im_ab, warp_ba, warp_aba, im_gbl_ab, bgr_mc_im]
    for l in range(5):
        layer = inputs[4 + l]
        err_ba = func(layer[0], *args, **kwargs)
        err_ab = func(layer[1], *args, **kwargs)

        layer_data.append([err_ba, err_ab])

    return layer_data


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        for t in self.transforms:
            inputs = t(inputs)
        return inputs


class Concatenate(object):
    """
    Input: [im_l, im_ab, inputs]
    inputs = [warp_ba_l, warp_ba_ab, warp_aba, err_pm, err_aba]

    Output:[im_l, err_pm, warp_ba, warp_aba, im_ab, err_aba]
    """
    def __call__(self, inputs):
        im_l     = inputs[0]
        im_ab    = inputs[1]
        warp_ba  = inputs[2]
        warp_aba = inputs[3]
        # im_glb_ab = inputs[4]
        # bgr_mc_im = inputs[5]
        # bgr_mc_im = bgr_mc_im[[2, 1, 0], ...]

        err_ba   = []
        err_ab   = []

        for l in range(5):
            layer = inputs[4 + l]
            err_ba.append(layer[0])
            err_ab.append(layer[1])

        cerr_ba   = torch.cat(err_ba,  0)
        cerr_ab   = torch.cat(err_ab,  0)

        return (im_l, cerr_ba, warp_ba, warp_aba, im_ab, cerr_ab)
        # return (im_l, cerr_ba, warp_ba, warp_aba, im_glb_ab, bgr_mc_im, im_ab, cerr_ab)


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, inputs):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        inputs = CustomFunc(inputs, F.to_mytensor)
        return inputs


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __call__(self, inputs):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """

        im_l  = F.normalize(inputs[0], 50, 1)  # [0, 100]
        im_ab = F.normalize(inputs[1], (0, 0), (1, 1))  # [-100, 100]

        inputs[2][0:1, :, :] = F.normalize(inputs[2][0:1, :, :], 50, 1)
        inputs[2][1:3, :, :] = F.normalize(inputs[2][1:3, :, :], (0, 0), (1, 1))
        warp_ba = inputs[2]

        inputs[3][0:1, :, :] = F.normalize(inputs[3][0:1, :, :], 50, 1)
        inputs[3][1:3, :, :] = F.normalize(inputs[3][1:3, :, :], (0, 0), (1, 1))
        warp_aba = inputs[3]

        # im_gbl_ab = F.normalize(inputs[4], (0, 0), (1, 1))  # [-100, 100]
        #
        # bgr_mc_im = F.normalize(inputs[5], (123.68, 116.78, 103.938), (1, 1, 1))

        # layer_data = [im_l, im_ab, warp_ba, warp_aba, im_gbl_ab, bgr_mc_im]
        layer_data = [im_l, im_ab, warp_ba, warp_aba]

        for l in range(5):
            layer = inputs[4 + l]
            err_ba = F.normalize(layer[0], 127, 2) #[0, 255]
            err_ab = F.normalize(layer[1], 127, 2) #[0, 255]
            layer_data.append([err_ba, err_ab])

        return layer_data


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, inputs):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        return CustomFunc(inputs, F.resize, self.size, self.interpolation)


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, inputs):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            inputs = CustomFunc(inputs, F.pad, self.padding)

        i, j, h, w = self.get_params(inputs[0], self.size)
        return CustomFunc(inputs, F.crop, i, j, h, w)


class CenterCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = (h - th) // 2
        j = (w - tw) // 2
        return i, j, th, tw

    def __call__(self, inputs):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            inputs = CustomFunc(inputs, F.pad, self.padding)

        i, j, h, w = self.get_params(inputs[0], self.size)
        return CustomFunc(inputs, F.crop, i, j, h, w)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, inputs):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """

        if random.random() < 0.5:
            return CustomFunc(inputs, F.hflip)
        return inputs


class RGB2Lab(object):
    def __call__(self, inputs):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """

    def __call__(self, inputs):
        image_lab = color.rgb2lab(inputs[0])
        warp_ba_lab = color.rgb2lab(inputs[2])
        warp_aba_lab = color.rgb2lab(inputs[3])
        # im_gbl_lab = color.rgb2lab(inputs[4])

        inputs[0] = image_lab[:, :, :1]     # l channel
        inputs[1] = image_lab[:, :, 1:]     # ab channel
        inputs[2] = warp_ba_lab             # lab channel
        inputs[3] = warp_aba_lab            # lab channel
        # inputs[4] = im_gbl_lab[:, :, 1:]    # ab channel

        return inputs