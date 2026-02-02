## How to a build component


1. Make a copy of this folder and name it whatever you want to call your component, also rename the python script and deployment yaml in the folder.  In all the following commands use you component name (instead of template_process).


2. Copy the orchestrate wrapper folder and any other required scripts/packages into the component folder.  The orchestrate wrapper and some general packages are located in the `general_libraries` folder.

```bash
cp -r ../../general_libraries/gfm_logger . && cp -r ../../general_libraries/gfm_data_processing . && cp -r ../../general_libraries/orchestrate_wrapper .
```

3. Use the CLAIMED library to build the component; this uses the current directory `pwd` as the docker context; ensure main script and additional files/dirs are in the same folder.  This will build a docker image starting from the `Dockerfile.template`.  It will include the 

```bash
c3_create_operator --repository quay.io/geospatial-studio --dockerfile_template_path Dockerfile.template --log_level DEBUG --version v0.1.0 --local_mode terrakit_data_fetch.py gfm_logger gfm_data_processing sentinelhub_config.toml
```

4. Push the image to a container registry from where it can be deployed:
```bash
docker push quay.io/geospatial-studio/template_process:v0.1.0
```

5. Remove gfm_logger and gfm_data_processing from current directory
```bash
rm -r  gfm_logger gfm_data_processing orchestrate_wrapper ./*.cwl ./*.job.yaml terrakit_data_fetch.yaml
```

## Deploy the process component
To deploy the component to OpenShift you will use the deployment script in the folder.  In the deployment script you will need to:

1. Update the name of the process (replace `pipeline-template_process` with `pipeline-{component name}`).
2. Add any extra environment variables which are required by the process and update `process_id` and `process_exec`.  All components will require:
    * `orchestrate_db_uri` - the uri to the orchestration db (holds the list of tasks)
    * `process_id` - the id of the process.  This will be used to search for available pipeline steps in tasks in the orchestration table.
    * `process_exec` - the command to run the component (e.g. `'python claimed_template_process.py'`).
3. When you are ready to deploy, log in to the cluster using `oc` in the terminal, then:
```bash
oc apply -f deploy_template_process.yaml
```