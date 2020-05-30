FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y curl \
        && apt-get install -y libeigen3-dev

RUN apt-get install -y caffe-cuda libcaffe-cuda-dev python3-opencv

ADD . /src/app

WORKDIR /src/app/similarity_subnet

RUN apt-get install -y libboost-dev libboost-system-dev libboost-filesystem-dev libgflags-dev

RUN apt-get install -y libgoogle-glog-dev libprotobuf-dev libopenblas-dev libopencv-dev

RUN /bin/bash scripts/make_binaries.sh

RUN echo "Downloading Weight Files..." && /bin/bash scripts/download_weights.sh

WORKDIR /src/app

RUN apt-get install -y python3-pip && pip3 install -r requirements.txt