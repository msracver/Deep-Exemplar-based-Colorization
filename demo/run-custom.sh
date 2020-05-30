#!/bin/bash

mkdir -p /src/app/custom-example/res
mkdir -p /src/app/custom-example/flow
mkdir -p /src/app/custom-example/combo_new

/src/app/demo/bin/deep_image_analogy /src/app/demo/models/deep_image_analogy/ /src/app/custom-example/ 0 2 0

/src/app/demo/bin/similarity_combo /src/app/demo/models/similarity_subnet/ /src/app/custom-example/ 0 2 0

python3 /src/app/colorization_subnet/test.py --short_size 512 --test_model /src/app/demo/models/colorization_subnet/example_net.pth --data_root /src/app/custom-example/ --out_dir /src/app/custom-example/res/ --gpu_id 0
