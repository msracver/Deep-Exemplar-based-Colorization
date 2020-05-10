#!/bin/bash

bin/deep_image_analogy models/deep_image_analogy/ example/ 0 2 0

bin/similarity_combo models/similarity_subnet/ example/ 0 2 0

python3 ../colorization_subnet/test.py --short_size 256 --test_model models/colorization_subnet/example_net.pth --data_root example/ --out_dir example/res/ --gpu_id 0
