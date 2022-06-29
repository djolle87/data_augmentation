FROM python:3.8-slim

USER root

SHELL ["/bin/bash", "-c"]

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:/src:$PATH"
ENV IMPORT_DATA_PATH=/data_augmentation/import_data
ENV EXPORT_DATA_PATH=/data_augmentation/export_data

RUN apt-get update && \
    apt-get install -y libsndfile1

WORKDIR /data_augmentation

VOLUME /src
VOLUME /data_augmentation/import_data
VOLUME /data_augmentation/export_data

COPY ./requirements.txt /data_augmentation/requirements.txt

RUN python3.8 -m venv $VIRTUAL_ENV && \
    source $VIRTUAL_ENV/bin/activate && \
    python3.8 -m pip install --no-cache-dir -r /data_augmentation/requirements.txt

CMD python3.8 -m src.routines.main