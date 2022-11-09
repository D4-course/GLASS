FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
# RUN apt-get update && apt-get upgrade -y
# RUN apt-get install python-pip -y
# RUN apt-get install wget curl -y
# SHELL ["/bin/bash","--login", "-c"] 
# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     # && https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp37-cp37m-linux_x86_64.whl \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 
# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"
# RUN echo ". /root/miniconda3/etc/profile.d/conda.sh" >> ~/.profile
# RUN conda init bash
# RUN conda create -n py396 python=3.9.6
# RUN echo "source activate py396" > ~/.bashrc
# RUN echo "Switched to Python 3.9.6"
RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3.9 -y

RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==1.7.2 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
RUN pip install pytest
RUN pip install optuna
RUN pip install pyyaml
# RUN pip install *.whl
# COPY requirements.txt .
# RUN cat requirements.txt 
# RUN pip install -r requirements.txt
# RUN conda install -c rdkit rdkit=2018.09.1
# RUN apt-get install libxrender1 -y
# RUN apt install -y libsm6 libxext6 libxrender-dev 
WORKDIR /app
COPY ./ ./
# WORKDIR /app/model
WORKDIR /app/
RUN echo "python3 GLASSTest.py" >> ~/.bashrc
