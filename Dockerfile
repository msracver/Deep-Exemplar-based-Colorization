FROM nvcr.io/nvidia/pytorch:19.06-py3

RUN apt-get update -y

RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh

RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b

RUN rm Miniconda-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:${PATH}

RUN conda update -y conda

RUN conda create -n torch python=3.6 -y

RUN echo "source activate torch" > ~/.bashrc

ENV PATH /opt/conda/envs/env/bin:$PATH


RUN mkdir -p /src/app
WORKDIR /src/app

ADD requirements.txt /src/app/requirements.txt
RUN source activate torch && conda install --file requirements.txt
ADD colorization_subnet /src/app/colorization_subnet

ADD demo /src/app/demo

ADD similarity_subnet /src/app/similarity_subnet

RUN mkdir -p /src/app/demo/models/colorization_subnet

RUN wget -O /src/app/demo/models/colorization_subnet/example_net.pth http://pretrained-models.auth-18b62333a540498882ff446ab602528b.storage.gra5.cloud.ovh.net/image/exemplar-colorization/example_net.pth

