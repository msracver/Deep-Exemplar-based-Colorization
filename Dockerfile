FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND=noninteractive

ADD . /src/app/

RUN apt-get update -y && apt-get install -y curl && \
    apt-get install -y build-essential && \
    apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev && \
    apt-get install -y unzip

WORKDIR /src/app/

# Build and install Open CV 3.4.10-dev
RUN apt-get install -y python3-pip && pip3 install --upgrade pip && pip3 install numpy

RUN curl -L -o opencv.zip https://github.com/opencv/opencv/archive/3.4.10.zip

RUN unzip opencv.zip && rm opencv.zip

RUN mkdir -p /src/app/opencv-3.4.10/build

WORKDIR /src/app/opencv-3.4.10/build

RUN cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D PYTHON3_EXECUTABLE=/usr/bin/python3 -D PYTHON_INCLUDE_DIR=/usr/include/python3.5m \
    -D PYTHON_INCLUDE_DIR2=/usr/include/x86_64-linux-gnu/python3.5m/ \
    -D PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so \
    -D PYTHON_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.5/dist-packages/numpy/core/include/ .. &&\
    make -j4 && make install

# Cleanup 
RUN rm -rf /src/app/opencv-3.4.10

# Build Caffe
RUN apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler

RUN apt-get install --no-install-recommends -y libboost-all-dev

RUN apt-get install -y libatlas-base-dev

RUN apt-get install -y libgoogle-glog-dev liblmdb-dev libeigen3-dev

WORKDIR /src/app/caffe

RUN make -j4 all

# Build Similarity Sub-network
WORKDIR /src/app/similarity_subnet

RUN /bin/bash scripts/make_binaries.sh

ENV LD_LIBRARY_PATH=/usr/local/lib:/src/app/caffe/build/lib:${LD_LIBRARY_PATH}

WORKDIR /src/app/

RUN pip3 install -r requirements.txt
