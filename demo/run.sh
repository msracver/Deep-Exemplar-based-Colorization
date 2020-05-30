#!/bin/bash

if [ "$#" -lt 1 ]; then
	EXAMPLE=example
else
	EXAMPLE=$1
fi

NUM_LINES=$(cat $EXAMPLE/pairs.txt|wc -l)

bin/deep_image_analogy models/deep_image_analogy/ $EXAMPLE 0 $NUM_LINES 0

bin/similarity_combo models/similarity_subnet/ $EXAMPLE 0 $NUM_LINES 0

python3 ../colorization_subnet/test.py --short_size 256 --test_model models/colorization_subnet/example_net.pth --data_root $EXAMPLE --out_dir $EXAMPLE/res/ --gpu_id 0

