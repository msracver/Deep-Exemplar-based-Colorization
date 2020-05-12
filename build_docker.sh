#!/bin/bash

./similarity_subnet/scripts/download_weights.sh
docker build -t deep-colorization .
