# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


FROM registry.access.redhat.com/ubi8/ubi-minimal AS base

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system and runtime dependencies
RUN true \
    && microdnf -y update \
    && microdnf install -y \
        mesa-libGL cmake tar gzip jq procps vi rsync \
    && microdnf clean all

#######################################################################################################
## Geospatial conda environment layer
# FROM base AS conda_geospatial_env
# Installs the Geospatial libraries into an isolated conda env

# Need gcc to install some dependencies
RUN microdnf install -y \
    gcc \
&& microdnf clean all

# Install conda and use the base environment and add link to pip in /usr/local/bin
RUN true \
&& cd /tmp \
&& curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-py311_25.11.1-1-Linux-x86_64.sh \
&& chmod +x Miniconda3-*-Linux-x86_64.sh \
&& bash ./Miniconda3-*-Linux-x86_64.sh -bf -p /opt/miniconda \
&& rm Miniconda3-*-Linux-x86_64.sh \
&& /opt/miniconda/bin/conda config --system --add channels conda-forge \
&& /opt/miniconda/bin/conda config --system --add channels nodefaults \
&& /opt/miniconda/bin/conda config --system --remove channels defaults \
&& /opt/miniconda/bin/conda config --system --set channel_priority strict \
&& /opt/miniconda/bin/conda install -y sqlite gdal libgdal libgdal-arrow-parquet h5netcdf \
&& /opt/miniconda/bin/conda clean -afy \
&& ln -s /opt/miniconda/bin/pip /usr/local/bin/pip \
&& /opt/miniconda/bin/conda update pip

ENV PATH=/opt/miniconda/bin:$PATH
ENV PROJ_LIB=/opt/miniconda/share/proj/
ENV USE_PYGEOS="0"


# # Build section from pipelines components
# FROM conda_geospatial_env AS pipeline_processor_image
USER root

RUN pip install --upgrade pip
RUN pip install ipython nbformat pg8000 sqlalchemy

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

RUN mkdir /pipeline

########## Adding core general libraries
COPY ./general_libraries/gfm_logger /pipeline/gfm_logger
COPY ./general_libraries/gfm_data_processing /pipeline/gfm_data_processing
COPY ./general_libraries/orchestrate_wrapper /pipeline/

########## Adding specific processor scripts
# inference planner
COPY ./components/inference-planner/inference_planner.py /pipeline/
COPY ./components/inference-planner/inference_planner_functions.py /pipeline/

# postprocess-generic-single
COPY ./components/postprocess-generic-single/postprocess-generic-single.py /pipeline/
COPY ./components/postprocess-generic-single/postprocess_generic_helper_functions.py /pipeline/
COPY ./components/postprocess-generic-single/postprocess_regularization.py /pipeline/

# push_to_geoserver
COPY ./components/push_to_geoserver/push_to_geoserver.py /pipeline/
COPY ./components/push_to_geoserver/push_to_geoserver_helper_functions.py /pipeline/

# run-inference
COPY ./components/run-inference/run-inference.py /pipeline/
COPY ./components/run-inference/inference_helper /pipeline/inference_helper

# terrakit_data_fetch
COPY ./components/terrakit_data_fetch/terrakit_data_fetch.py /pipeline/
COPY ./components/terrakit_data_fetch/sentinelhub_config.toml /pipeline/

# url-connector-single
COPY ./components/url-connector-single/url_connector_single.py /pipeline/
COPY ./components/url-connector-single/preprocessing_helper /pipeline/preprocessing_helper

# curated_upload_v2
COPY ./components/curated_upload/claimed_curated_upload_v2.py /pipeline/


######## Group ownership, data directory and file permission
RUN mkdir /pipeline/data \
&& mkdir -p /.config/sentinelhub \
&& mkdir /.docker && chmod -R 777 /.docker \
&& mkdir /.local && chmod -R 777 /.local \
&& mkdir /.cache && chmod -R 777 /.cache \
&& chown -R 10001:10001 /pipeline/ \
&& chmod -R 777 /pipeline/

COPY ./components/terrakit_data_fetch/sentinelhub_config.toml /.config/sentinelhub/config.toml
RUN chmod -R 777 /.config/sentinelhub/

ENV SH_PROFILE=cdse

USER 10001
WORKDIR "/pipeline/"

ADD ./general_libraries/orchestrate_wrapper/orchestrate_wrapper.py /pipeline/

CMD ["sh", "-c", "python /pipeline/orchestrate_wrapper.py"]

