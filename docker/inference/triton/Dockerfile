ARG PYTHON_VERSION=3.9.13
FROM 007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:23.03-py3

# PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK - see https://github.com/autogluon/autogluon/issues/2534
ENV PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1

# Specify accept-bind-to-port LABEL for inference pipelines to use SAGEMAKER_BIND_TO_PORT
# https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipeline-real-time.html
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
# Specify multi-models LABEL to indicate container is capable of loading and serving multiple models concurrently
# https://docs.aws.amazon.com/sagemaker/latest/dg/build-multi-model-build-container.html
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"


ENV PYTHONPATH="/opt/tritonserver/lib/python3.8/site-packages"

RUN pip install --no-cache-dir -U --trusted-host pypi.org --trusted-host files.pythonhosted.org pip

# Install AutoGluon

ARG AUTOGLUON_VERSION=0.7.0

RUN pip install --no-cache-dir -U --trusted-host pypi.org --trusted-host files.pythonhosted.org pip \
 && pip install --no-cache-dir -U setuptools wheel \
 && pip uninstall -y dataclasses \
 && pip install --no-cache-dir -U numpy numba \
 && pip install --no-cache-dir -U autogluon==${AUTOGLUON_VERSION} \
 && mim install -q mmcv-full \
 && pip install --no-cache-dir -U mmdet


# RUN curl -o /licenses-autogluon.txt https://autogluon.s3.us-west-2.amazonaws.com/licenses/THIRD-PARTY-LICENSES.txt
RUN pip install tritonclient[all]
# install dependencies in the local requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt
EXPOSE 8000 8001
ENTRYPOINT ["/usr/bin/python3.8", "/code/src/main.py", "triton"] 
# ENTRYPOINT tritonserver --model-repository /opt/ml/model --allow-sagemaker true --log-error true --log-warning true --log-info true
