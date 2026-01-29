# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


FROM registry.access.redhat.com/ubi8/ubi-minimal as base

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
FROM base as conda_geospatial_env
# Installs the Geospatial libraries into an isolated conda env

# Need gcc to install some dependencies
RUN microdnf install -y \
    gcc \
&& microdnf clean all

# Install conda and use the base environment and add link to pip in /usr/local/bin
RUN true \
&& cd /tmp \
&& curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-py311_25.9.1-3-Linux-x86_64.sh \
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
