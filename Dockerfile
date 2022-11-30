FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3.9 -y
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==1.7.2 -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
RUN pip install pytest optuna pyyaml uvicorn streamlit==1.9.2 fastapi
WORKDIR /app
COPY ./ ./
WORKDIR /app/