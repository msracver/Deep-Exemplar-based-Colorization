#!/bin/bash

curl -L -C - -o demo/models/similarity_subnet/vgg_19_gray_bn/vgg19_bn_gray_ft_iter_150000.caffemodel https://www.dropbox.com/s/liz78q1lf9bc57s/vgg19_bn_gray_ft_iter_150000.caffemodel?dl=0 &&

mkdir -p demo/models/colorization_subnet
curl -L -C - -o demo/models/colorization_subnet/example_net.pth https://www.dropbox.com/s/rg6qi5iz3sj7cnc/example_net.pth?dl=0 &&

mkdir -p demo/models/deep_image_analogy/vgg19
curl -L -C - -o demo/models/deep_image_analogy/vgg19/VGG_ILSVRC_19_layers.caffemodel http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel &&

echo Downloadiing Completed!
