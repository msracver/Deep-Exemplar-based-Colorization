#!/bin/bash

mkdir -p /src/app/demo/bin

# Make Similarity Combo
SOURCE=/src/app/similarity_subnet/linux/similarity_combo/source

nvcc $SOURCE/*.cpp $SOURCE/*.cu -I/src/app/caffe/include -I/src/app/caffe/build/src -L/src/app/caffe/build/lib -o /src/app/demo/bin/similarity_combo -std=c++11\
    -I/usr/include/eigen3 -lopencv_core -lopencv_highgui -lopencv_imgproc \
    -lopencv_imgcodecs -lboost_system -lboost_filesystem -lcublas -lcaffe -lglog \
    -Wno-deprecated-gpu-targets

# Make Deep Image Analogy
SOURCE=/src/app/similarity_subnet/linux/deep_image_analogy/source

nvcc $SOURCE/*.cpp $SOURCE/*.cu -I/src/app/caffe/include -I/src/app/caffe/build/src -L/src/app/caffe/build/lib -o /src/app/demo/bin/deep_image_analogy -std=c++11\
    -I/usr/include/eigen3 -lopencv_core -lopencv_highgui -lopencv_imgproc \
    -lopencv_imgcodecs -lboost_system -lcublas -lcaffe -lglog \
    -Wno-deprecated-gpu-targets
