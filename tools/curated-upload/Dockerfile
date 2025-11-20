# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


FROM registry.access.redhat.com/ubi9/python-311
USER root
ADD claimed_curated_upload_v2.py /opt/app-root/src/

RUN pip install --upgrade pip
RUN pip install ipython nbformat
RUN pip install --no-cache-dir rio-cogeo wget scikit-learn>=1.3.0 rasterio boto3==1.35.82 botocore numpy>=1.22.2 tqdm requests humanize terrakit
RUN chmod -R 777 /opt/app-root/src/
USER default
WORKDIR "/opt/app-root/src/"
CMD ["python", "claimed_curated_upload_v2.py"]