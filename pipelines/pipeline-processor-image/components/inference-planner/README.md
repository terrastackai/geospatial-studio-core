

## inference-planner component


### Installing pre-requisites for building the component
- Claimed. The component is built using claimed compiler. To install claimed [see docs](https://github.com/romeokienzler/c3?tab=readme-ov-file#install)

- Docker. Used to generate the docker images.


### How to build component


1. Copy orchestrate_wrapper to current directory where `inference-planner.py` lives

```
cp -r ../../general_libraries/gfm_logger . && cp -r ../../general_libraries/gfm_data_processing . && cp -r ../../general_libraries/orchestrate_wrapper .

```

2. Create component using c3; uses current directory `pwd` as the docker context; ensure main script and additional files/dirs are in the same folder

```
c3_create_operator --repository quay.io/geospatial-studio --dockerfile_template_path Dockerfile.template  --log_level DEBUG --version v0.1.0 --local_mode inference_planner.py inference_planner_functions.py gfm_logger gfm_data_processing sentinelhub_config.toml
```

3. Remove orchestrate_wrapper from current directory
```
rm -r  gfm_logger gfm_data_processing orchestrate_wrapper ./*.cwl ./*.job.yaml inference_planner.yaml
```

### Further Works
This component can be adapted for other modalities e.g. sentinelhub, openeo, etc. Just needs to replace the initial part of fetching the data.
