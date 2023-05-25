FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:0.7.0-gpu-py39-cu117-ubuntu20.04

# set the working directory
WORKDIR /code

RUN apt-get -y update
RUN apt-get -y install git
# install dependencies in the local requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# install autogluon
RUN pip3 install -U pip
RUN pip3 install -U setuptools wheel

RUN git clone https://github.com/autogluon/autogluon.git
RUN cd autogluon && chmod +x ./CI/docker/full_install_image.sh && ./CI/docker/full_install_image.sh
RUN mim install mmcv-full --timeout 60
RUN pip install --upgrade mmdet
RUN pip install --upgrade mmocr

ENTRYPOINT ["python", "/code/src/main.py", "tabular"] 
