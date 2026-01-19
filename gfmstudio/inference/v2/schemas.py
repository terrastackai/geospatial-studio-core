# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from ...common.schemas import ItemResponse, ListResponse


# ***************************************************
# Model
# ***************************************************
class ModelOnboardingInputSchema(BaseModel):
    fine_tuned_model_id: str = Field(description="", default=None, max_length=100)
    model_configs_url: str = Field(
        description="Presigned url model config file.", default=None
    )
    model_checkpoint_url: str = Field(
        description="Presigned url to model checkpoint file.", default=None
    )

    class Config:
        from_attributes = True
        protected_namespaces = ()


class ModelUpdateInput(BaseModel):
    display_name: str
    description: Optional[str] = None
    model_url: Optional[HttpUrl] = None
    pipeline_steps: Optional[List[Dict[str, Any]]] = None
    geoserver_push: Optional[List[Dict[str, Any]]] = None
    model_input_data_spec: Optional[List[Dict[str, Any]]] = None
    postprocessing_options: Optional[Dict] = None
    sharable: Optional[bool] = False
    model_onboarding_config: Optional[ModelOnboardingInputSchema] = None
    latest: Optional[bool] = None

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "model display_name can only contain letters, underscores, and hyphens"
            )
        return v


class ModelCreateInput(ModelUpdateInput):
    version: Optional[float] = 1.0


class ModelCreate(ModelCreateInput):
    internal_name: Optional[str] = Field(
        None,
        description="Internal name for the model, generated automatically based on display_name.",
    )
    latest: Optional[bool] = True


class ModelGetResponse(ItemResponse, ModelCreate):
    status: Optional[str]


class ModelListResponse(ListResponse):
    results: Optional[List[ModelGetResponse]] = []


# ***************************************************
# Inference
# ***************************************************
class SpatialDomain(BaseModel):
    bbox: Optional[List[List[float]]] = Field(default_factory=list)
    polygons: Optional[List] = Field(default_factory=list)
    tiles: Optional[List] = Field(default_factory=list)
    urls: Optional[List] = Field(default_factory=list)

    @model_validator(mode="after")
    def at_least_one_field_required(cls, model):
        if not (model.bbox or model.polygons or model.tiles or model.urls):
            raise ValueError(
                "At least one of 'bbox', 'polygons', 'tiles', or 'urls' must be provided."
            )
        return model


class DataSource(BaseModel):
    connector: Optional[str] = None
    collection: Optional[str] = None
    bands: Optional[Union[List[Dict[str, Any]]] | Dict] = None
    scaling_factor: Optional[List[float]] = None
    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    def update_missing_fields(cls, data):
        if ("collection_name" in data) and ("collection" not in data):
            data["collection"] = data["collection_name"]
        return data


class PostProcessing(BaseModel):
    cloud_masking: Optional[Union[bool, str, dict]] = None
    snow_ice_masking: Optional[Union[bool, str, dict]] = None
    permanent_water_masking: Optional[Union[bool, str, dict]] = None
    ocean_masking: Optional[Union[bool, str, dict]] = None
    model_config = {"extra": "allow"}


class GeoServerPush(BaseModel):
    workspace: str
    layer_name: str
    display_name: str
    filepath_key: str
    file_suffix: str
    geoserver_style: Union[str, dict]
    model_config = {"extra": "allow"}


class InferenceConfig(BaseModel):
    spatial_domain: SpatialDomain
    temporal_domain: List[str] = None
    model_input_data_spec: Optional[List[Dict[str, Any]]] = None
    data_connector_config: Optional[List[DataSource]] = None
    geoserver_push: Optional[List[GeoServerPush]] = None
    pipeline_steps: Optional[List[Dict[str, Any]]] = None
    post_processing: Optional[PostProcessing] = None
    fine_tuning_id: Optional[str] = None
    maxcc: Optional[int] = 100


class ModelOnboardingConfig(BaseModel):
    fine_tuned_model_id: Optional[str] = None
    model_configs_url: str = None
    model_checkpoint_url: str = None
    model_framework: Optional[str] = None


class InferenceWebhookMessageDetails(BaseModel):
    message: Optional[str] = None
    error: Optional[str] = None
    status: Optional[str] = None


class InferenceWebhookMessages(BaseModel):
    timestamp: datetime
    detail: Optional[InferenceWebhookMessageDetails] = {}


class InferenceCreate(BaseModel):
    description: Optional[str] = None
    location: Optional[str] = None
    inference_config: InferenceConfig
    geoserver_layers: Optional[Dict[str, Any]] = None
    priority: Optional[str] = None
    queue: Optional[str] = None
    demo: Optional[Dict[str, Any]] = None
    status: Optional[str] = "PENDING"
    model_id: Optional[UUID] = None
    inference_output: Optional[Dict[str, Any]] = None

    model_config = {"extra": "ignore"}


class InferenceCreateInput(InferenceConfig):
    model_display_name: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    geoserver_layers: Optional[Dict[str, Any]] = None
    # priority: Optional[str] = None
    # queue: Optional[str] = None
    fine_tuning_id: Optional[str] = None
    demo: Optional[Dict[str, Any]] = None
    model_id: Optional[UUID] = None
    inference_output: Optional[Dict[str, Any]] = None
    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def check_model_required(cls, model):
        if not (model.model_id or model.model_display_name):
            raise ValueError(
                "At least one of 'model_id' or 'model_display_name' must be provided."
            )
        return model


class InferenceGetResponse(ItemResponse, InferenceCreateInput):

    spatial_domain: SpatialDomain
    temporal_domain: List[str] = None
    model_input_data_spec: Optional[List[Dict[str, Any]]] = Field(
        default=None, exclude=True
    )
    data_connector_config: Optional[List[DataSource]] = Field(
        default=None, exclude=True
    )
    geoserver_push: Optional[List[GeoServerPush]] = Field(default=None, exclude=True)
    pipeline_steps: Optional[List[Dict[str, Any]]] = Field(default=None, exclude=True)
    post_processing: Optional[PostProcessing] = Field(default=None, exclude=True)
    status: Optional[str]
    updated_at: Optional[datetime] = None
    tasks_count_total: Optional[int] = None
    tasks_count_success: Optional[int] = None
    tasks_count_failed: Optional[int] = None
    tasks_count_stopped: Optional[int] = None
    tasks_count_waiting: Optional[int] = None
    fine_tuning_id: Optional[str] = None
    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def populate_inference_config(cls, values):

        if isinstance(values, dict) and values["inference_config"]:
            values["spatial_domain"] = values["inference_config"]["spatial_domain"]
            values["temporal_domain"] = values["inference_config"]["temporal_domain"]
            values["model_display_name"] = values["model"].display_name
            values["fine_tuning_id"] = values["inference_config"]["fine_tuning_id"]
        else:
            values.spatial_domain = values.inference_config["spatial_domain"]
            values.temporal_domain = values.inference_config["temporal_domain"]
            values.model_display_name = values.model.display_name
            values.fine_tuning_id = values.inference_config.get("fine_tuning_id")

        return values


class InferenceGetResponseWebhooks(InferenceGetResponse):
    model_config = {"extra": "ignore"}


class InferenceListResponse(ListResponse):
    results: Optional[List[InferenceGetResponse]] = []


# ***************************************************
# Task
# ***************************************************
class TaskCreate(BaseModel):
    inference_id: UUID
    task_id: Optional[str] = None
    inference_folder: Optional[str] = None
    status: Optional[str] = "PENDING"
    pipeline_steps: Optional[List[Dict[str, Any]]] = None

    class Config:
        from_attributes = True


class TaskGetResponse(BaseModel):
    inference_id: UUID
    task_id: Optional[str] = None
    inference_folder: Optional[str] = None
    status: Optional[str] = "PENDING"
    pipeline_steps: Optional[List[Dict[str, Any]]] = None
    updated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class InferenceTasksListResponse(BaseModel):
    inference_id: UUID
    status: str
    tasks: List[TaskCreate]

    class Config:
        from_attributes = True


# ***************************************************
# DataSource
# ***************************************************
# class DataSourceCreateInput(BaseModel):
#     data_connector: Optional[str] = None
#     collection_id: Optional[str] = None
#     data_connector_config: Optional[Dict[str, Any]] = None
#     sharable: Optional[bool] = False


class DataSourceGetResponse(BaseModel):
    connector: str
    collection_name: str
    bands: Union[Dict, List]
    data_collection: Optional[str] = None
    resolution_m: Optional[int] = None
    rgb_bands: Optional[List] = None
    modality_tag: Optional[str] = None
    cloud_masking: Optional[Dict] = None
    query_template: Optional[str] = None
    search: Optional[Dict] = None
    request_input_data: Optional[Dict] = None
    model_config = {"extra": "allow"}


class DataSourceListResponse(BaseModel):
    results: Optional[List[DataSourceGetResponse]] = []


class DataAdvisorRequestSchema(BaseModel):
    collections: list[str] = None
    dates: list[str] = None
    bbox: Optional[list[list[float]]] = None
    area_polygon: Optional[str] = None
    maxcc: Optional[float] = None
    pre_days: int = 1
    post_days: int = 1


# ***************************************************
# Notification
# ***************************************************
class NotificationCreate(BaseModel):
    event_id: UUID = Field(
        description="An identifier generated for every event when an inference job is started."
    )
    detail_type: str = Field(
        description="Describes the nature of the event.",
        example="Inference:Task:Notifications",
    )
    source: str = Field(
        description="Identifies the service that generated the webhook event.",
        example="com.ibm.prithvi-100m-hls2-flood-segmentation",
    )
    timestamp: datetime = Field(
        description="The event timestamp specified by the service generating the event.",
        example="2023-08-13T16:31:47Z",
    )
    detail: Dict[Any, Any] = Field(
        description="JSON object with information about the event."
    )
    inference_id: Optional[UUID] = None


class NotificationGetResponse(BaseModel):
    id: Optional[UUID] = None
    detail: Optional[InferenceWebhookMessageDetails] = {}
    source: Optional[str] = None
    detail_type: Optional[str] = None
    timestamp: Optional[datetime] = None
    event_id: UUID = Field(
        description="An identifier generated for every event when an inference job is started."
    )

    class Config:
        from_attributes = True


class NotificationListResponse(ListResponse):
    results: list[NotificationGetResponse] = []


# ***************************************************
# V2 Pipelines
# ***************************************************
class V2PipelineCreate(InferenceConfig):
    model_id: str
    model_internal_name: str
    tune_id: Optional[str] = None
    request_type: Optional[str] = "openeo"
    inference_id: str
    description: Optional[str] = None
    location: Optional[str] = None
    model_access_url: Optional[str] = None
    user: Optional[str] = None

    model_config = {"extra": "allow"}


class FilesShareOut(BaseModel):
    """Response Model for file-share get endpoint."""

    upload_url: str
    download_url: str
    message: str


# ***************************************************
# Generic Processor component task
# ***************************************************
class GenericProcessorCreate(BaseModel):
    """Generic Processor Create Input Schema."""
    name: str
    description: Optional[str] = None
    processor_parameters: Optional[Dict[str, Any]] = None

class GenericProcessorGetResponse(ItemResponse, GenericProcessorCreate):
    """Generic Processor Get Response Schema."""
    status: Optional[str]   
    name: Optional[str] = None
    description: Optional[str] = None
    processor_file_path: Optional[str] = None
    processor_parameters: Optional[Dict[str, Any]] = None
    processor_presigned_url: Optional[str] = None

    class Config:
        from_attributes = True

    
class GenericProcessorListResponse(ListResponse):
    """Generic Processor List Response Schema."""
    results: Optional[List[GenericProcessorGetResponse]] = []
    