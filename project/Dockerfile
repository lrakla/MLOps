# FROM python:3.8-slim

# WORKDIR /project

# COPY requirements_inference.txt requirements_inference.txt

# RUN pip install --no-cache-dir torch==2.3.1 -r requirements_inference.txt --extra-index-url https://download.pytorch.org/whl/cpu

# RUN rm -rf /root/.cache

# COPY . .

# RUN mkdir models

# RUN python download_model.py

# # After deployment to lambda function
# CMD ["lambda_handler.lambda_handler"]

# For testing on local
# RUN python lambda_handler.py
FROM public.ecr.aws/lambda/python:3.8

WORKDIR ${LAMBDA_TASK_ROOT}

COPY . ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir torch==2.3.1 -r requirements_inference.txt --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements_inference.txt

ARG CACHE_DIR=/tmp/transformers_cache
ARG MODEl_DIR=models

RUN mkdir $CACHE_DIR
RUN mkdir $MODEl_DIR

ENV TRANSFORMERS_CACHE=${MODEL_DIR} \
    TRANSFORMERS_VERBOSITY=error

RUN python load_model.py

CMD ["lambda_function.lambda_handler"]