#!/bin/bash

mkdir -p /src/app/custom-example/res
mkdir -p /src/app/custom-example/flow
mkdir -p /src/app/custom-example/combo_new

bin/similarity_combo models/similarity_subnet/ /src/app/custom-example/ 0 2 0

python3 ../colorization_subnet/test.py --short_size 256 --test_model models/colorization_subnet/example_net.pth --data_root /src/app/custom-example/ --out_dir /src/app/custom-example/res/ --gpu_id 0
