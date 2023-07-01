# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT=3-bullseye
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN sed -i 's/archive\.ubuntu\.com/au.archive.ubuntu.com/g' /etc/apt/sources.list

# Copy library scripts to execute
COPY .devcontainer/library-scripts/*.sh .devcontainer/library-scripts/*.env /tmp/library-scripts/

# [Option] Install zsh
ARG INSTALL_ZSH="true"
# [Option] Upgrade OS packages to their latest versions
ARG UPGRADE_PACKAGES="true"
# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    # Remove imagemagick due to https://security-tracker.debian.org/tracker/CVE-2019-10131
    && apt-get purge -y imagemagick imagemagick-6-common \
    # Install common packages, non-root user
    && bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# # Setup default python tools in a venv via pipx to avoid conflicts
# ENV PIPX_HOME=/usr/local/py-utils \
#     PIPX_BIN_DIR=/usr/local/py-utils/bin
# ENV PATH=${PATH}:${PIPX_BIN_DIR}
# RUN bash /tmp/library-scripts/python-debian.sh "none" "/usr/local" "${PIPX_HOME}" "${USERNAME}" \ 
#     && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
ARG NODE_VERSION="none"
ENV NVM_DIR=/usr/local/share/nvm
ENV NVM_SYMLINK_CURRENT=true \
    PATH=${NVM_DIR}/current/bin:${PATH}
RUN bash /tmp/library-scripts/node-debian.sh "${NVM_DIR}" "${NODE_VERSION}" "${USERNAME}" \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Remove library scripts for final image
RUN rm -rf /tmp/library-scripts

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1



#FROM mcr.microsoft.com/devcontainers/python:0-3.10
#FROM nvidia/cuda:11.8-devel-ubuntu20.04
# eog is a great way to get x11 deps on
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install eog wget ninja-build python3.10 python3-pip -y && \
    rm -rf /var/lib/apt/lists/*

#COPY .devcontainer/install_cuda.sh /tmp/pip-tmp/
#RUN  /tmp/pip-tmp/install_cuda.sh

#

RUN pip3 install --upgrade pip

COPY requirements.txt /tmp/pip-tmp/

RUN /usr/bin/python3 -m pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

