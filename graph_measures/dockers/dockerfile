FROM nvidia/cuda:10.1-runtime-ubuntu18.04

ENV CONDA_PATH=/opt/anaconda3
ENV ENVIRONMENT_NAME=boost
SHELL ["/bin/bash", "-c"]

COPY . /tools

# curl is required to download Anaconda.
RUN apt-get update && apt-get install curl wget -y

# Download and install Anaconda.
RUN cd /tmp && curl -O https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
RUN chmod +x /tmp/Anaconda3-2019.07-Linux-x86_64.sh
RUN mkdir /root/.conda
RUN bash -c "/tmp/Anaconda3-2019.07-Linux-x86_64.sh -b -p ${CONDA_PATH}"

# Initializes Conda for bash shell interaction.
RUN ${CONDA_PATH}/bin/conda init bash

# Upgrade Conda to the latest version
RUN ${CONDA_PATH}/bin/conda update -n base -c defaults conda -y

# Create the work environment and setup its activation on start.
RUN ${CONDA_PATH}/bin/conda create --name ${ENVIRONMENT_NAME} python=3.7  -y
RUN echo conda activate ${ENVIRONMENT_NAME} >> /root/.bashrc

RUN . ${CONDA_PATH}/bin/activate ${ENVIRONMENT_NAME} \
 && conda env update --file /tools/boost.yml --prune

#RUN apt-get update
#RUN conda install pip
#RUN pip install tensorboard_logger
#WORKDIR /eli

