FROM mcr.microsoft.com/devcontainers/python:0-3.10
#FROM nvidia/cuda:11.8-devel-ubuntu20.04
# eog is a great way to get x11 deps on
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install eog wget -y && \
    rm -rf /var/lib/apt/lists/*

COPY .devcontainer/install_cuda.sh /tmp/pip-tmp/
RUN  /tmp/pip-tmp/install_cuda.sh

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /tmp/pip-tmp/

RUN /usr/local/bin/python3 -m pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

