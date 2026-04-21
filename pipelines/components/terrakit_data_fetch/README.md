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

## Caching Feature

The terrakit_data_fetch component now includes intelligent caching to significantly improve performance for repeated queries.

### How It Works

1. **Cache Check**: Before fetching from Terrakit, checks if data exists in cache
2. **Cache Hit**: Copies (or hardlinks) files from `/data/cache` to task folder (~2-5 seconds)
3. **Cache Miss**: Fetches from Terrakit, processes, and caches results (~30-60 seconds)
4. **Performance**: 10-30x faster for repeated queries with same spatial-temporal parameters

### Configuration

Add these environment variables to your `values.yaml`:

```yaml
- name: TERRAKIT_CACHE_ENABLED
  value: "true"
- name: TERRAKIT_CACHE_DIR
  value: "/data/cache"  # Uses existing shared PVC
- name: TERRAKIT_CACHE_TTL_DAYS
  value: "30"
- name: TERRAKIT_CACHE_MAX_SIZE_GB
  value: "400"  # Optional size limit
```

See `cache_config_example.yaml` for complete configuration example.

### Benefits

- ✅ **10-30x faster** for repeated queries
- ✅ **Reduced API costs** (fewer Terrakit calls)
- ✅ **Better reliability** (cached data always available)
- ✅ **Shared across pods** (all processors can access cache)
- ✅ **Automatic cleanup** (size-based eviction)
- ✅ **Zero infrastructure changes** (uses existing PVC)

### Management

```bash
# Monitor cache size
kubectl exec -it <pod-name> -- du -sh /data/cache

# Clear cache
kubectl exec -it <pod-name> -- rm -rf /data/cache/*

# Disable cache temporarily
kubectl set env deployment/terrakit-data-fetch TERRAKIT_CACHE_ENABLED=false
```

For detailed documentation, see `../../general_libraries/terrakit_cache/README.md`

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