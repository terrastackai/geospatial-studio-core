## Sentinelhub Inference Preprocessing Component


### Installing pre-requisites for building the component
- Claimed. The component is built using claimed compiler. To install claimed [see docs](https://github.com/romeokienzler/c3?tab=readme-ov-file#install)

- Docker. Used to generate the docker images.


### How to build component


1. Copy gfm_logger and gfm_data_processing to current directory where `sentinelhub_connector.py` lives

```
cp -r ../../general_libraries/gfm_logger . && cp -r ../../general_libraries/gfm_data_processing . && cp -r ../../general_libraries/orchestrate_wrapper .

```

2. Create component using c3; uses current directory `pwd` as the docker context; ensure main script and additional files/dirs are in the same folder

```
c3_create_operator --repository quay.io/geospatial-studio --dockerfile_template_path Dockerfile.template --log_level DEBUG --version v0.1.0 --local_mode url_connector_single.py gfm_logger gfm_data_processing preprocessing_helper
```

3. Remove gfm_logger and gfm_data_processing from current directory
```
rm -r gfm_logger gfm_data_processing orchestrate_wrapper ./*.cwl ./*.job.yaml url_connector_single.yaml
```
