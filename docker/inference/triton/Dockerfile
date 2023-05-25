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


# RUN apt-get update \
#  && apt-get -y upgrade \
#  && apt-get autoremove -y \
#  && apt-get install tesseract-ocr -y \
#  && apt-get clean \
#  && rm -rf /var/lib/apt/lists/*

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

# Removing GluonTS nursery/tsbench package - it is not used in training/inference and have security vulnerabilities
RUN rm -rf /usr/local/lib/python3.9/dist-packages/gluonts/nursery/tsbench

# Remove these to address security issues; these are not used if only python package is used
RUN rm -rf /usr/local/lib/python3.9/dist-packages/ray/jars \
 && rm -rf /opt/conda/lib/python3.9/site-packages/ray/jars

# RUN HOME_DIR=/root \
#  && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
#  && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
#  && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
#  && chmod +x /usr/local/bin/testOSSCompliance \
#  && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
#  && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} python \
#  && rm -rf ${HOME_DIR}/oss_compliance*

# RUN curl -o /licenses-autogluon.txt https://autogluon.s3.us-west-2.amazonaws.com/licenses/THIRD-PARTY-LICENSES.txt
RUN pip install tritonclient[all]
RUN pip install typer[all]
EXPOSE 8000 8001
ENTRYPOINT ["/usr/bin/python3.8", "/code/src/main.py", "triton"] 
# ENTRYPOINT tritonserver --model-repository /opt/ml/model --allow-sagemaker true --log-error true --log-warning true --log-info true
