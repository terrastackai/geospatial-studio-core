![Geospatial Studio banner](./docs/images/banner.png)

# 🌍 GEO Studio: Pipelines

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

[![TerraTorch](https://img.shields.io/badge/TerraTorch-a3b18a)](https://github.com/terrastackai/terratorch)
[![TerraKit](https://img.shields.io/badge/TerraKit-588157)](https://github.com/terrastackai/terrakit)
[![Iterate](https://img.shields.io/badge/Iterate-3a5a40)](https://github.com/terrastackai/iterate)
[![CLAIMED](https://img.shields.io/badge/CLAIMED%20Framework-c3)](https://github.com/claimed-framework)

![Helm](https://img.shields.io/badge/Helm-0F1689?style=flat&logo=helm)
![Red Hat OpenShift](https://img.shields.io/badge/-Red_Hat_OpenShift-EE0000?logo=redhatopenshift&logoColor=white)
![Kubernetes](https://img.shields.io/badge/kubernetes-326CE5?&logo=kubernetes&logoColor=white)
![Postgresql](https://img.shields.io/badge/PostgreSQL-316192?logo=postgresql&logoColor=white)
![Minio](https://img.shields.io/badge/-MinIO-C72E49?logo=minio&logoColor=white)

[![Studio Documentation](https://img.shields.io/badge/Studio_Documentation-526CFE?style=for-the-badge&logo=MaterialForMkDocs&logoColor=white)](https://terrastackai.github.io/geospatial-studio)

---

## 🚀 Overview
The pipelines engine of the Geospatial Studio is a collection of modular processing steps which can be orchestrated to pull and prepare data to input for AI models, run inference (with deployed or ephemeral models) and customisable post-processing steps.

When an inference request is received, an inference planning step will search for available data and plan a series of tasks which, generally, split the overall workflow by sub-bounding boxes and dates, which can then be run in parallel.

```mermaid
---
config:
  theme: neutral
  layout: dagre
---
flowchart LR
 subgraph subGraph0["Pipeline process"]
        D["Inference Planner"]
  end
 subgraph subGraph1["Pipeline subtask 0"]
        F["TerraKit data pull"]
        G["Run inference"]
        H["Generic post-processing"]
        I["Push to Geoserver"]
  end
 subgraph subGraph2["Pipeline subtask 1"]
        J["TerraKit data pull"]
        K["Run inference"]
        L["Generic post-processing"]
        M["Push to Geoserver"]
  end
 subgraph subGraph3["Pipeline subtask 2"]
        N["TerraKit data pull"]
        O["Run inference"]
        P["Generic post-processing"]
        Q["Push to Geoserver"]
  end
    A(["Inference request"]) --> B["Gateway API"]
    B --> C["PlannerTask"]
    C --> D
    D --> E["PipelineTasks"]
    E --> F & J & N
    F --> G
    G --> H
    H --> I
    J --> K
    K --> L
    L --> M
    N --> O
    O --> P
    P --> Q
    D@{ shape: subproc}
    B@{ shape: div-proc}
    C@{ shape: db}
    E@{ shape: db}
    style D fill:#00C853,stroke:#000000
    style F fill:#00C853,stroke:#000000
    style G fill:#00C853,stroke:#000000
    style H fill:#00C853,stroke:#000000
    style I fill:#00C853,stroke:#000000
    style J fill:#00C853,stroke:#000000
    style K fill:#00C853,stroke:#000000
    style L fill:#00C853,stroke:#000000
    style M fill:#00C853,stroke:#000000
    style N fill:#00C853,stroke:#000000
    style O fill:#00C853,stroke:#000000
    style P fill:#00C853,stroke:#000000
    style Q fill:#00C853,stroke:#000000
    style A fill:#E1BEE7,stroke:#000000
    style B fill:#BBDEFB,stroke:#000000
    style C fill:#FFE0B2,stroke:#000000
    style E fill:#FFE0B2,stroke:#000000
    style subGraph0 fill:#C8E6C9,stroke:#000000
    style subGraph3 fill:#C8E6C9,stroke:#000000
    style subGraph2 fill:#C8E6C9,stroke:#000000
    style subGraph1 fill:#C8E6C9,stroke:#000000
```

Each pipeline component is a python script built as a container, which includes the code to pick up tasks from the orchestration database.  The components are deployed as microservices on the k8s/Red Hat Openshift and can be scaled up and down to support changing workloads, with different components scale independently dependent of their resource requirements.

When idle, the deployed pipeline components will constantly check for new tasks which it could pick up.  Tasks can be assigned priority level to move up or down the queue.  Functionality to deploy different queues is also possible, but not yet implemented in the Geospatial Studio.


<!-- ## Features -->




## 🔀 Pipeline orchestration

The pipelines run through a series of microservices which are deployed on the openshift cluster and can be scaled up and down (currently manual) depending on the load.  Each task have a series of pipeline steps, defined in the following way:
```json
[
  {"status":"READY","process_id":"terrakit-data-fetch","step_number":0},
  {"status":"WAITING","process_id":"run-inference","step_number":1},
  {"status":"WAITING","process_id":"postprocess-generic","step_number":2},
  {"status":"WAITING","process_id":"push-to-geoserver","step_number":3}
]
```

Within the microservices there is the core processing code (mainly python scripts), which is executed by the `orchestrate_wrapper.py` script.  

This script:
1. checks the [task table](#inference-task-table) (in the orchestration db) for any tasks where it's `process_id` is in the `READY` state.
2. * if there are no tasks ready, it will wait and try again a few seconds later (#1).
   * if it finds a task waiting, it will change the status to `RUNNING`, then execute the process code (e.g. to query data, run inferece).  
3. * Once the code finishes, the wrapper will update the status of that step in the DB to `FINISHED` if successful, then mark the next step as `READY` and it should get picked up by the next process.  
   * If the code errors, it should mark the step as `FAILED` and the pipeline should stop.

```mermaid
---
config:
  layout: elk
  look: neo
---
flowchart LR
 subgraph subGraph0["Orchestrate wrapper"]
        B{"Is there a READY task?"}
        D["Wait 8 seconds"]
  end
 subgraph subGraph1["Orchestrate wrapper"]
        E{"Did it complete successfully?"}
        F["Mark step FINISHED and next step READY"]
        G["Mark step FAILED"]
        H["Available"]
  end
    subGraph0 ~~~ subGraph1
    B -- Yes --> C["Execute processor code"]
    B -- No --> D
    D --> B
    C --> E
    E -- Yes --> F
    E -- No --> G
    F --> H
    G --> H
    H --> B
    F@{ shape: event}
    G@{ shape: event}
    classDef fix line-height:20px
    style B fill:#BBDEFB,color:#000000
    style D fill:#FFE0B2,color:#000000
    style E fill:#E1BEE7,color:#000000
    style F color:#000000,fill:#C8E6C9
    style G fill:#FFCDD2,color:#000000
    style H fill:#BBDEFB,color:#000000
    style subGraph0 fill:#FFF9C4
    style subGraph1 fill:#FFF9C4
    style C fill:#00C853,color:#000000
```


Functionality will be added in the near future to:
* enable fan-in/fan-out in pipelines.





## 💻 Getting started

If you don't have access to a deployed version of the Geospatial Studio, the easiest way to get started with the pipelines is to deploy a version locally, this is particularly useful for developing new pipeline components.

--> [Deployment instructions](https://github.com/terrastackai/geospatial-studio)

## 💠 Existing pipeline componenets

|        Name    |                  Description                    |
|      :------  | :----------------------------------------------|
| Planner  | Orchestrates tasks by breaking down the inference job based on spatio-temporal queries |
| Terrakit data pull | Acquires remote sensing data required for inferencing. Uses data connectors such as nasa earthdata, sentinel aws, sentinelhub, etc |
| Url connector| Process user provided geospatial data |
| Run inference | Execute inference on a detached inferencing service|
| Terratorch inference  | Utilizes terratorch to perform model predictions |
| Postprocess generic | Masks prediction results; denoting features such as cloud cover, permanent water, ice, snow etc
| Push to geospatial store | Uploads the input and prediction results to a geospatial store for sharing and visualization |

## 🛠 Developing new pipeline components

To build a process component, you can use CLAIMED to build the container image from the python script.  You will likely need to read the [inference configuration](components/template_process/inference_config_template.json) and [task configuration](components/template_process/task_config_template.json) from the folder and an example of how to do that can be seen in the [template process script](components/template_process/template_process.py).

The process script should read the configurations and possibly data output from the previous pipeline steps (catalogued in the task config file).  It should then update the task config at the end to include details of any new layers/data products calculated.

### 🧱 Steps to build a new component

Follow the instructions in the [template process](components/template_process).  That folder provides a template for creating a new process component.

<!-- 1. Create a new folder in the components directory (in theory could be done anywhere).
2. Create a python script which will contain the process code.  There is a template file which can be used which includes the required headers for CLAIMED, importing the core inference pipeline environment variables, reading the inference and task configs and adding information to the task config at the end of the run.
3. Copy the `Dockerfile.template` into the component folder
4. Copy the `orchestrate_wrapper` folder into the component folder
5. Copy any other required libraries/scripts into the component folder -->

## 📜 Appendix

### Inference Task Table

Snippet of an inference task with task_id 6f031be7-86b5-4128-b23e-857dfcc65a25-task_0

This task has two subtasks: url-connector and push-to-geoserver.

| pipeline_steps | status | inference_id | task_id | inference_folder | priority | queue |
| :------        | :----- | :----------- | :------ | :--------------- | :------  | :---- |
| [{"status":"RUNNING","process_id":"url-connector","step_number":0}, {"status":"READY","process_id":"push-to-geoserver","step_number":1}] | RUNNING | 6f031be7-86b5-4128-b23e-857dfcc65a25 | 6f031be7-86b5-4128-b23e-857dfcc65a25-task_0 | /data/6f031be7-86b5-4128-b23e-857dfcc65a25 | 5 | |


