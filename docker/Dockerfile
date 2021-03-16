FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo /home/user/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh \
    && chmod +x /home/user/miniconda.sh \
    && /home/user/miniconda.sh -b -p /home/user/miniconda \
    && rm /home/user/miniconda.sh \
    && conda install -y python==3.7 \
    && conda clean -ya

# Install PyTorch
RUN conda install -y -c pytorch \
    cudatoolkit=10.0 \
    "pytorch=1.4.0=py3.7_cuda10.0.130_cudnn7.6.3_0" \
    "torchvision=0.5.0=py37_cu100" \
    && conda clean -ya

# Install tensorflow
RUN pip install tensorflow-gpu==2.0.0

# Install Jupyter Notebook
RUN conda install -y -c conda-forge jupyterlab

# selecting a work dir also creates the directory, so we use it to create /data as a mounting point
WORKDIR /data

# our actual working directory
WORKDIR /workingdir

#make port 8888 available from outside, just in case the container is supposed to run jupyter notebooks
EXPOSE 8888
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Adding a directories with permissions for matplotlib for Docker-support
WORKDIR /.cache
RUN chmod 777 /.cache

WORKDIR /.config/matplotlib
RUN chmod 777 /.config/matplotlib

WORKDIR /.local
RUN chmod 777 /.local


# Adding dlbio as this is a core-requirement
# If you wish to be able to see into the dlbio code, move the dlbio folder
# into the docker fodler and use the file as is. Otherwise remove the three
# lines below and uncomment the three after 'WORKDIR /workingdir'.
#WORKDIR /dlbio
#COPY './dlbio' /dlbio
#RUN pip install -e /dlbio

# Adding the workingdirectory
WORKDIR /workingdir

RUN pip install git+https://github.com/pgruening/dlbio
RUN apt update
RUN apt install libgl1-mesa-glx -y

# Installing the remaining Phase Reconstruction requirements
RUN pip install opencv-python-headless \
    pytest-shutil \
    -U scikit-learn \
    seaborn \
    segmentation_models_pytorch \
    tqdm \
    utils \
    git+https://github.com/cybertronai/pytorch-lamb.git \
    openpyxl \
    xlrd 
#glob3 \

RUN pip install Pillow

# adjusting the matplotlib config for Docker-support
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc