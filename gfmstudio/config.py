# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import ConfigDict, Field, PostgresDsn
from pydantic_settings import BaseSettings

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
SENTINEL_SECRET_VALUE = "PLACEHOLDER-SECRET-MUST-BE-REPLACED"


class Settings(BaseSettings):
    DATABASE_URI: Optional[PostgresDsn] = None
    TEST_DATABASE_URI: Optional[str] = None
    AUTH_DATABASE_URI: Optional[PostgresDsn] = None
    REDIS_URL: Optional[str] = "redis://localhost:6379/0"

    DEBUG: bool = False
    ENVIRONMENT: Optional[str] = "local"  # Options local,
    CELERY_TASKS_ENABLED: Optional[bool] = Field(
        default=False,
        description="Whether to enable celery tasks to run FT tasks in the background.",
    )

    EIS_API_KEY: Optional[str] = ""

    # Object storage / COS details
    OBJECT_STORAGE_ENDPOINT: Optional[str] = Field(
        description="COS endpoint", default=""
    )
    OBJECT_STORAGE_KEY_ID: str = Field(description="Key ID for COS authentication")
    OBJECT_STORAGE_SEC_KEY: str = Field(description="Secret Key for COS authentication")
    OBJECT_STORAGE_REGION: Optional[str] = Field(
        description="URL with the region", default=""
    )
    OBJECT_STORAGE_SIGNATURE_VERSION: Optional[str] = Field(default="s3v4")

    TEMP_UPLOADS_BUCKET: Optional[str] = Field(
        description="Temporary uploads bucket name",
        default="geospatial-studio-temporary-uploads",
    )
    # Add pipelines v2 COS credentials
    PIPELINES_V2_COS_BUCKET: Optional[str] = Field(
        default="test-geo-inference-pipelines"
    )
    PIPELINES_V2_INFERENCE_ROOT_FOLDER: Optional[str] = Field(default=None)
    PIPELINES_V2_INTEGRATION_TYPE: Optional[str] = Field(
        default="database"
    )  # Options: database, kafka, api

    DEFAULT_SYSTEM_USER: Optional[str] = "system@ibm.com"
    AUTH_ENABLED: bool = Field(
        default=True,
        description="Whether authentication is turned on or off for the current environment.",
    )
    OAUTH_ENDPOINT: str = Field(default="")
    OAUTH_CLIENTID: str = Field(default="")
    OAUTH_CLIENTSECRET: str = Field(default="")

    AUTO_MODEL_ONBOARDING_BASE_URL: Optional[str] = Field(
        description="Base URL to the automated model onboarding service.",
        default="",
    )

    # V1 Pipelines
    INFERENCE_PIPELINE_V2_BASE_URL: Optional[str] = Field(
        default="https://pipeline-inference-api-nasageospatial.cash.sl.cloud9.ibm.com/"
    )
    INFERENCE_PIPELINE_BASE_URL: Optional[str] = Field(
        default="https://pipelines-orchestration-nasageospatial-uat.cash.sl.cloud9.ibm.com/v1"
    )
    INFERENCE_PIPELINE_ID: Optional[str] = Field(
        default="23a3e4e9-d81d-4694-a2b2-543581e63c12"
    )
    DEPLOY_FOR_INFERENCE_PIPELINE_ID: Optional[str] = Field(
        default="ad249995-58ed-4d56-9ec4-41021f75ee23"
    )

    DATA_ADVISOR_ENABLED: Optional[bool] = Field(default=False)
    DATA_ADVISOR_MAX_CLOUD_COVER: Optional[float] = Field(default=80)
    DATA_ADVISOR_PRE_DAYS: Optional[int] = Field(default=1)
    DATA_ADVISOR_POST_DAYS: Optional[int] = Field(default=1)

    # API Key Encryption
    API_ENCRYPTION_KEY: str = Field(default=SENTINEL_SECRET_VALUE)

    # Rate Limiting
    RATELIMIT_ENABLED: Optional[bool] = False  # Turn rate limit on/off
    RATE_LIMIT_CONFIG: Optional[dict] = {}
    RATELIMIT_LIMIT: Optional[int] = Field(
        default=200,
        desctiption="Overral requests limit (limit/window). Set 0 to turn off",
    )
    RATELIMIT_WINDOW: Optional[int] = Field(
        default=60,
        desctiption="Overral requests window (limit/window). Set 0 to turn off",
    )
    RATELIMIT_SENSITIVE_RESOURCE_LIMIT: Optional[int] = Field(
        default=6,
        desctiption="Sensitive resource endpoint requests limit (limit/window). Set 0 to turn off",
    )
    RATELIMIT_SENSITIVE_RESOURCE_WINDOW: Optional[int] = Field(
        default=300,
        desctiption="Sensitive resource endpoint requests window (limit/window). Set 0 to turn off",
    )

    # JIRA Configs
    JIRA_ISSUE_TYPES_PARENTS: dict = Field(
        description="Jira issuetypes and parent ids",
        default={
            "Story": "WGS-1145",
            "Bug": "WGS-1144",
            "Change Request": "WGS-1143",
            "Feature": "WGS-1142",
            "Incident": "WGS-1141",
            "Risk": "WGS-1137",
            "Service Ticket": "WGS-1136",
            "Task": "WGS-1135",
        },
    )
    JIRA_API_KEY: Optional[str] = Field(
        description="Jira API Key with write access", default=""
    )
    JIRA_API_URI: str = Field(
        description="Jira API URI",
        default="https://jsw.ibm.com/rest/api/2",
    )
    JIRA_API_PROJECT: str = Field(description="Jira API Project", default="WGS")
    JIRA_API_FIELDS: str = Field(
        description="Jira API Fields to be returned",
        default="id,key,comment,issuetype,created,updated,duedate,status,summary,description",
    )

    ####################
    # FINE TUNING
    ####################
    FT_IMAGE_PULL_SECRETS: str = Field(
        default="ris-private-registry", description="Image pull secret to pull images."
    )
    MMSEGMENTATION_GEO_IMAGE: str = Field(
        default="us.icr.io/gfmaas/mmsegmentation_geospatial:v0.1.0",
        description="The mmsegmentation docker image to run the fine tune process",
    )
    FT_HPO_IMAGE: Optional[str] = Field(default="us.icr.io/gfmaas/gfmstudio-hpo:v1.0.7")

    # Add temporary files upload COS credentials
    # Data Configs
    DATA_PVC: Optional[str] = Field(description="", default="gfm-ft-data-pvc")
    TUNES_FILES_BUCKET: Optional[str] = Field(
        description="COS bucket where the files are stored",
        default="geoft-service",
    )
    DATASET_FILES_BUCKET: Optional[str] = Field(
        default="geodev-dataset-factory", description="Dataset factory COS bucket"
    )
    #
    # Define mount points for PVCs
    #
    DATA_MOUNT: str = Field(description="", default="/data")
    FILES_MOUNT: str = Field(
        description="Path in the pod where the files PVC is mounted",
        default="/geotunes/",
    )
    BACKBONE_MODELS_MOUNT: str = Field(
        description="Path in the pod where the backbone models PVC is mounted",
        default="/terratorch/",
    )
    FILES_PVC: Optional[str] = Field(
        description="Name of the Persistent Volume ", default="gfm-ft-files-pvc"
    )
    NAMESPACE: str = Field(
        default="geoft",
        description="This is the namespace (or OCP project) where the helm upgrade is run",
    )

    TUNE_BASEDIR: str = "/files"

    LOGLEVEL: str = Field(description="Logging level", default="INFO")

    MIN_CHECKPOINT_SIZE: int = Field(
        description="Minimum size in bytes for the checkpoint files found in COS buckets",
        default=(1024**2) * 20,  # minimum 20Mb
    )

    CONFIG_FILE_TYPES: list[str] = Field(
        description="Config files in order of preference for fine tuning",
        default=("yaml", "yml", "py"),
    )

    # Model Deployment configs
    FT_API_KEY: Optional[str] = Field(
        description="API Key to authenticate the Fine tuning api.",
        default="empty",
    )
    SATOKEN: Optional[str] = Field(default="")

    MLFLOW_URL: Optional[str] = Field(
        description="URL to internal deployed version of Mlflow",
        default="https://gfm-mlflow-internal-nasageospatial-dev.cash.sl.cloud9.ibm.com",
    )
    GEOFT_WEBHOOK_URL: Optional[str] = Field(
        description="URL for webhook notifications",
        default="https://geoft-api-internal-nasageospatial-dev.cash.sl.cloud9.ibm.com/v2/notifications",
    )
    RUN_TERRATORCH_TEST: Optional[bool] = Field(
        description="Whether to run terratorch test after finetuning is complete",
        default=True,
    )

    # FineTuning image Configs
    RESOURCE_LIMIT_CPU: Optional[int] = 10
    RESOURCE_LIMIT_Memory: Optional[int] = 32
    RESOURCE_LIMIT_GPU: Optional[int] = 1
    RESOURCE_REQUEST_CPU: Optional[int] = 6
    RESOURCE_REQUEST_Memory: Optional[int] = 24
    RESOURCE_REQUEST_GPU: Optional[int] = 1
    CONFIGURE_GPU_AFFINITY: Optional[bool] = Field(
        default=True,
        description=(
            "Enable or disable GPU node affinity when scheduling fine-tuning jobs. "
            "If set to False, the job can run on any available GPU in the cluster. "
            "Turn this off if all GPUs are acceptable for fine-tuning, or if the "
            "user is not testing end-to-end fine-tuning."
        ),
    )
    NODE_SELECTOR_KEY: Optional[str] = Field(
        default="nvidia.com/gpu.product",
        description=(
            "The node label key used for GPU node affinity. By default, this uses "
            "the standard Kubernetes label `nvidia.com/gpu.product` provided by "
            "the NVIDIA device plugin."
        ),
    )
    NODE_GPU_SPEC: Optional[str] = Field(
        default="NVIDIA-A100-SXM4-80GB",
        description=(
            "Comma-separated list of GPU types that can be used for fine-tuning. "
            "For example: 'NVIDIA-A100-SXM4-80GB,NVIDIA-V100-SXM2-32GB'. "
            "If multiple values are given, the pod can be scheduled on any node "
            "matching one of them."
        ),
    )

    JOB_MAX_RETRY_COUNT: Optional[int] = Field(
        default=30,
        deacription="Maximum retry attempts to check if tuning job is complete. Defaults to 30 retries",
    )
    KJOB_MAX_WAIT_SECONDS: Optional[int] = Field(
        default=600,
        deacription="Max time to wait for tuning job to complete. Defaults to 10mins",
    )
    TERRATORCH_V2_START_DATE: Optional[str] = Field(
        default="2025-02-19",
        description="Cut-off data after which terratorch v2 should be in use",
    )

    ####################
    # DATASET FACTORY
    ####################
    # Data pipeline configs
    DATA_PIPELINE_BASE_URL: Optional[str] = Field(
        default="https://geofm-workflow-orchestrator-internal-nasageospatial-dev.cash.sl.cloud9.ibm.com/v1"
    )
    DATA_ONBOARD_PIPELINE_ID: Optional[str] = Field(
        default="3148e9aa-ee1e-40ba-a9a7-8e783deac6b7"
    )
    DATA_ONBOARD_PIPELINE_V2_ID: Optional[str] = Field(
        default="335d21ce-269e-47d5-91eb-65f315b5c728"
    )
    DATASET_PIPELINE_IMAGE: Optional[str] = Field(
        default="us.icr.io/gfmaas/geostudio-curated-upload:latest"
    )
    model_config = ConfigDict(
        extra="allow", case_sensitive=True, env_file=os.path.join(BASE_DIR, ".env")
    )

    ####################
    # AMO
    ####################
    SERVICE_ACCOUNT_NAME: Optional[str] = Field(default="geostudio-sa")
    AMO_RESOURCE_NAME: Optional[str] = Field(default="geospatial-dev")
    CONFIGURE_INFERENCE_TOLERATION: Optional[str] = Field(default="noToleration")
    AMO_API_PVC_ACCESS_MODE: Optional[str] = Field(default="ReadWriteMany")
    GATEWAY_SECRET_NAME: Optional[str] = Field(default="geofm-gateway-secrets")
    GFMAAS_API_CRED_KEY: Optional[str] = Field(default="FT_API_KEY")
    AMO_API_PVC_STORAGE_CAPACITY: Optional[str] = Field(default="20Gi")
    AMO_API_PVC_STORAGE_CLASS: Optional[str] = Field(default="ibmc-s3fs-cos-perf")
    AMO_API_MODEL_ARTIFACTS_DOCKER_IMAGE_URL: Optional[str] = Field(
        default="us.icr.io/gfmaas/model-artifacts-download:v0.0.1-alpha"
    )
    INFERENCE_SVC_CONTAINER_IMAGE: Optional[str] = Field(
        default="us.icr.io/gfmaas/geospatial-model-inference-service:main-579"
    )
    INFERENCE_SVC_TERRATORCH_V2_IMAGE: Optional[str] = Field(
        default="us.icr.io/gfmaas/geospatial-model-inference-service:main-579"
    )
    COS_CREDENTIALS_SECRET_NAME: Optional[str] = Field(default="studio-cos-secret")
    AMO_FILES_BUCKET: Optional[str] = Field(
        description="COS bucket where new model artifacts are stored",
        default="geodev-amo-input-bucket",
    )
    AMO_INFERENCE_SHARED_PVC: Optional[str] = Field(
        description="Inference Shared PVC", default="inference-shared-pvc"
    )


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings()


settings = get_settings()
