FORM nvcr.io/nvidia/pytorch:19.06-py3

RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh

RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b

RUN rm Miniconda-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:${PATH}

RUN conda update -y conda

RUN conda install opencv

RUN conda install -c pytorch pytorch

RUN conda install -c pytorch torchvision

RUN conda install opencv

WORKDIR /src

ADD colorization_subnet /src/

ADD demo /src/

ADD similarity_subnet /src/

RUN wget -O /src/demo/models/colorization_subnet/example_net.pth http://pretrained-models.auth-18b62333a540498882ff446ab602528b.storage.gra5.cloud.ovh.net/image/exemplar-colorization/example_net.pth

