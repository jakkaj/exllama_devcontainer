#!/bin/bash

#wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb
dpkg -i cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb
cp /var/cuda-repo-debian11-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
add-apt-repository contrib
apt-get update
apt-get -y install cuda
#rm cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb