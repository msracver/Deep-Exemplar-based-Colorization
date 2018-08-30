# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import torch.utils.data as data

from PIL import Image
import os
import struct
import os.path as osp
import numpy as np
import cv2

def parse_images(dir):
	dir = osp.expanduser(dir)
	image_pairs = []
	pair_file = osp.join(dir, 'pairs.txt')
	if osp.exists(pair_file):
		with open(pair_file, "r") as f:
			for line in f:
				pair = line.strip().split(" ")
				if len(pair) >=2 :
				    item0 = (pair[0], pair[1])
				    image_pairs.append(item0)
	else:
		raise (RuntimeError("Found no pair.txt in folder of: " + dir+ "\n"))
	return image_pairs

def pil_loader(path):
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

def combo5_loader(path, real_w, real_h):
	f = open(path, 'rb')

	# width, height
	d = f.read(4)
	im_sz = struct.unpack("i", d)
	h = im_sz[0]

	d = f.read(4)
	im_sz = struct.unpack("i", d)
	w = im_sz[0]

	# warp_ba_layer 4
	d = f.read(4)
	im_sz = struct.unpack("i", d)
	d = f.read(im_sz[0])
	file_bytes = np.asarray(bytearray(d), dtype=np.uint8)
	img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
	img_data_ndarray = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
	warp_ba = Image.fromarray(img_data_ndarray)

	# warp_aba_layer 4
	d = f.read(4)
	im_sz = struct.unpack("i", d)
	d = f.read(im_sz[0])
	file_bytes = np.asarray(bytearray(d), dtype=np.uint8)
	img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
	img_data_ndarray = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
	warp_aba = Image.fromarray(img_data_ndarray)

	# 5 layers: err_aba, err_ba, err_ab
	errs = []
	for l in range(5):
		d = f.read(4)
		im_sz = struct.unpack("i", d)
		d = f.read(im_sz[0])
		file_bytes = np.asarray(bytearray(d), dtype=np.uint8)
		img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
		err_ba = Image.fromarray(img_data_ndarray)

		d = f.read(4)
		im_sz = struct.unpack("i", d)
		d = f.read(im_sz[0])
		file_bytes = np.asarray(bytearray(d), dtype=np.uint8)
		img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
		err_ab = Image.fromarray(img_data_ndarray)
		errs.append([err_ba, err_ab])

	f.close()

	return errs, warp_ba, warp_aba


class TestDataset(data.Dataset):
	def __init__(self, data_root, transform=None):
		image_pairs = parse_images(data_root)
		if len(image_pairs) == 0:
			raise (RuntimeError("Found 0 image pairs in dataroot"))

		self.data_root = data_root
		self.image_pairs = image_pairs
		self.transform = transform

	def get_out_name(self, index):
		img_name0, img_name1 = self.image_pairs[index]
		out_name = '%s_%s.png' % (os.path.splitext(img_name0)[0], os.path.splitext(img_name1)[0])
		return out_name

	def __getitem__(self, index):
		image_id = 0
		pair_id = index

		image_names = ["", ""]
		image_names[0], image_names[1] = self.image_pairs[pair_id]

		image_path = osp.join(self.data_root, "input", image_names[image_id])
		image = pil_loader(image_path)

		inputs = [image, image]
		w, h = image.size

		image_comb_name = "%s_%s" % (os.path.splitext(image_names[image_id])[0], os.path.splitext(image_names[1 - image_id])[0])
		combo_path = osp.join(self.data_root, "combo_new", "%s.combo" % image_comb_name)
		errs, warp_ba, warp_aba = combo5_loader(combo_path, w, h)

		inputs.append(warp_ba)
		inputs.append(warp_aba)
		inputs = inputs + errs

		if self.transform is not None:
			inputs = self.transform(inputs)

		return inputs

	def __len__(self):
		return len(self.image_pairs)
