# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


FROM registry.access.redhat.com/ubi9/python-311:latest as base

WORKDIR /downloads

USER root

FROM base as miniforge

# Define the Miniforge installation path
ENV MAMBA_ROOT_PREFIX=/opt/miniforge

# Install Miniforge (faster Conda alternative)
RUN curl -LfsSo /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && chmod +x /tmp/Miniforge3.sh \
    && bash /tmp/Miniforge3.sh -bf -p $MAMBA_ROOT_PREFIX \
    && rm -f /tmp/Miniforge3.sh

ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

RUN mamba install -y -c conda-forge \
    libspatialindex gdal libgdal libgdal-arrow-parquet "urllib3<2" cudatoolkit=11.8 h5netcdf \
    # Ensure cryptography >= 43.0.1 is installed, resolves: https://github.com/pyca/cryptography/security/advisories/GHSA-h4gh-qq45-vh27
    && mamba install -y 'cryptography>=43.0.1'\
    && mamba clean -afy


FROM base as runtime_dependencies
# copy miniconda to path
COPY --from=miniforge /opt/miniforge/ /opt/miniforge/
ENV PATH=/opt/miniforge/bin:$PATH


# Download and install kubectl
RUN true && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x ./kubectl && \
    mv kubectl /usr/local/bin/ && \
    mkdir ~/.kube && \
    chmod 777 ~/.kube

WORKDIR /terratorch

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache \
    pip3 install -U pip && \
    # Install from main with terramind tiled_inference fix
    pip install terratorch==1.1 
    # pip install torchgeo==0.6.2 Remove this patch fix

FROM runtime_dependencies as dev_base
# Create a non-root user with a UID and GID greater than 10000
RUN groupadd -g 10001 appgroup && \
    useradd -r -u 10001 -g appgroup -ms /bin/bash appuser && \
    chown -R appuser:appgroup /terratorch

# Setup UTF-8 locale
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LC_CTYPE=C.UTF-8

USER appuser

# Image default command
CMD ["/bin/bash"]
