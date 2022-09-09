FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV TZ=Europe/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app
COPY . /app/

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    wget \
    curl \
    python3.7 \
    python3-pip \
    python3-dev \
    python3-setuptools

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

RUN pip3 install --upgrade setuptools pip
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html

ENTRYPOINT ["python3"]