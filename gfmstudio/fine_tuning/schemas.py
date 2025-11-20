# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import enum
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from gfmstudio.common.schemas import ItemResponse, ListResponse
from gfmstudio.config import settings
from gfmstudio.fine_tuning.core.schema import ModelBaseParams
from gfmstudio.inference.v2.schemas import DataSource, GeoServerPush, SpatialDomain


class TuneOptionEnum(str, enum.Enum):
    """TuneOpton Enum

    Attributes
    ----------
    K8_JOB : str
        If the Tune option is a Kubernetes Job
    MCAD : str
        If the Tune option is an MCAD Job
    RAY_IO : str
        If the Tune option is a Ray.io Job
    """

    K8_JOB = "k8_job"
    RAY_IO = "ray_io"


# ***************************************************
# Tunes Schemas
# ***************************************************
class TuneSubmitBase(BaseModel):
    """Model for the Tune submissions.

    Attributes
    ----------
    name : str
        Alphanumeric, no special characters or spaces name of the Tune with a min of 4 and max of 40 characters .
    description : str, Optional
        The description of the tune. Defaults to None.
    dataset_id: str
        The dataset to be used for the tuning job.
    base_model_id:
        A unique identifier for the model, generated using `uuid.uuid4()`.
    task_id:
        A unique identifier for the task, generated using `uuid.uuid4()`.
    model_parameters : dict, optional
        Tune parameters. Defaults to empty dictionary.
    train_options : dict, optional
       Training options for the Tune. Defaults to empty dictionary.

    Methods
    -------
    validate_name()
        Validates that the tune name is correct as expected.

    Raises
    ------
    ValueError
        If any required fields are not provided or if they contain invalid values.


    Examples
    --------
    >>> tune_submission = TuneSubmitIn(name = 'test', dataset_id = '123', base_model_id='123')
    >>> tune_submission.name
    'test'
    """

    name: str = Field(
        description="Alphanumeric, no special characters or spaces",
        min_length=4,
        max_length=30,
    )
    description: Optional[str] = None
    dataset_id: str

    class Config:
        protected_namespaces = ()

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        """Validates the tune name

        Parameters
        ----------
        name : str
            The name of the tune.

        Returns
        -------
        str
            Cleaned up name of the tune without special characters or white spaces.

        Raises
        ------
        ValueError
             If `name` contains special characters or white spaces.
        """
        # Clean-up the tune name.
        name = name.replace(" ", "-").replace("_", "-").strip()
        if not re.match("^[a-zA-Z0-9]+([.-]{0,1}[a-zA-Z0-9]+)*$", name):
            raise ValueError(
                "must not contain special characters or white spaces. Replace underscores with hyphens."
            )
        return name


class TuneSubmitIn(TuneSubmitBase):
    """Schema for tune submission."""

    base_model_id: Optional[uuid.UUID] = None
    tune_template_id: uuid.UUID
    model_parameters: Optional[Any] = {}
    train_options: Optional[Dict] = Field(
        description="Define options for training",
        default={},
    )


class TuneSubmitHPO(TuneSubmitBase):
    """Schema for hpo tune submission."""


class TuneTask(BaseModel):
    """Model for the Tune Task

    Attributes
    ----------
    id : uuid.UUID
        The unique identifier of the TuneTask, generated using `uuid.uuid4()`.
    name : str
        The name of the TuneTask

    Raises
    ------
    ValueError
        If any required fields are not provided or if they contain invalid values.

            Examples
    --------
    >>> tunes_task = TuneTask(id="123",name="some_name")
    >>> tunes_task.name
    'some_name'
    """

    id: uuid.UUID
    name: str


class TuneBaseModel(BaseModel):
    """Model for the Tune BaseModel

    Attributes
    ----------
    id : uuid.UUID
        The unique identifier of the TuneBaseModel, generated using `uuid.uuid4()`.
    name : str
        The name of the TuneBaseModel

    Raises
    ------
    ValueError
        If any required fields are not provided or if they contain invalid values.

    Examples
    --------
    >>> tunes_base_model = TuneBaseModel(id="123",name="some_name")
    >>> tunes_base_model.name
    'some_name'
    """

    id: uuid.UUID
    name: str


class TuneOut(ItemResponse):
    """
    Response model for tuning jobs.

    Attributes
    ----------
    id : str
        A unique identifier for the tuning job.
    name : str
        The name of the tuning job.
    description : str
        A description of the tuning job.
    task : TuneTask, optional
        The associated task for the tuning job. Defaults to None.
    dataset : TuneDataset, optional
        The dataset used for the tuning job. Defaults to None.
    base_model : TuneBaseModel, optional
        The base model used for the tuning job. Defaults to None.
    mcad_id : str, optional
        Identifier for the MCAD job. Defaults to an empty string. TODO: Change this to k8_job_id
    status : str, optional
        The current status of the tuning job. Defaults to an empty string.
    latest_chkpt : str, optional
        The filename of the latest checkpoint for the tuning job. Defaults to an empty string.

    Raises
    ------
    ValueError
        If any required fields are not provided or if they contain invalid values.

    Examples
    --------
    >>> tune_job = TuneOut(id='123', name='Tuning Job 1', description='Description of the job')
    >>> tune_job.status
    ''
    """

    id: str
    name: str
    description: str
    task: Optional[TuneTask] = None
    dataset_id: str
    base_model: Optional[TuneBaseModel] = None
    mcad_id: Optional[str] = ""
    status: Optional[str] = ""
    latest_chkpt: Optional[str] = ""
    logs: Optional[str] = ""
    metrics: Optional[str] = ""
    shared: Optional[bool] = False

    @model_validator(mode="before")
    def update_shared(cls, values):
        if isinstance(values, dict):
            values["shared"] = settings.DEFAULT_SYSTEM_USER == values["created_by"]
        else:
            values.shared = settings.DEFAULT_SYSTEM_USER == values.created_by
        return values

    class Config:
        from_attributes = True


class TuneStatusOut(TuneOut):
    """
    Response model for tuning job status.

    Attributes
    ----------
    config_json : dict, optional
        JSON configuration for the tuning job. Defaults to an empty dictionary.
    progress : dict, optional
        Progress of the tuning job. Defaults to None and is validated before use.
    metrics : list of dict, optional
        Metrics related to the tuning job. Defaults to an empty list and is validated before use.
    logs : Any, optional
        Logs associated with the tuning job. Defaults to None.
    logs_presigned_url : str, optional
        Presigned COS URL for accessing logs. Defaults to None.
    tuning_config: str, optional
        Tuning config associated with the tuning job. Defaults to None.
    tuning_config_presigned_url: str, optional
        Presigned COS URL for accessing the tuning config used for the tune. Defaults to None.
    tune_template_id:
        A unique identifier for the tune template.
    model_parameters : dict, optional
        Tune parameters. Defaults to empty dictionary.

    Raises
    ------
    ValueError
        If the `progress` or `metrics` fields cannot be parsed into valid JSON objects.

    Examples
    --------
    >>> tune_status = TuneStatusOut(
    ...     id='123',
    ...     name='Tuning Job 1',
    ...     description='Description of the job',
    ...     progress='{"total_iterations": 10, "complete_iterations": 5}',
    ...     metrics='[{"accuracy": 0.85}]'
    ... )
    >>> tune_status.progress
    {'total_iterations': 10, 'complete_iterations': 5}
    >>> tune_status.metrics
    [{'accuracy': 0.85}]
    """

    config_json: Optional[dict] = {}
    progress: Optional[dict] = None
    metrics: Optional[list[dict]] = []
    logs: Optional[Any] = None
    logs_presigned_url: Optional[str] = None
    tuning_config: Optional[str] = None
    tuning_config_presigned_url: Optional[str] = None
    train_options: Optional[dict] = {}
    tune_template_id: Optional[uuid.UUID] = None
    model_parameters: Optional[Any] = {}

    @field_validator("progress", mode="before")
    def update_progress(cls, val):
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except json.decoder.JSONDecodeError:
                val = {"error": "could not render tune progress"}

        if not val:
            val = {
                "total_iterations": 1,
                "complete_iterations": 0,
                "progress_fraction": 0,
            }
        return val

    @field_validator("metrics", mode="before")
    def update_metrics(cls, val):
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except json.decoder.JSONDecodeError:
                val = [{"error": "could not render tune metrics"}]
        return val


class TunesOut(ListResponse):
    """Response Model for Tunes list

    Attributes
    ----------
    ListResponse : list of TuneOut
        List of tunes

    Examples
    --------
    >>> tunes_response = TunesOut(results=[TuneOut(id='1', name='Tune Job 1', description='First tuning job')])
    >>> tunes_response.results[0].name
    'Tune Job 1'
    """

    results: list[TuneOut]


class TuneUpdateIn(BaseModel):
    """
    Model for updating tuning job details.

    Attributes
    ----------
    name : str, optional
        The name of the tuning job. Defaults to None.
    description : str, optional
        A description of the tuning job. Defaults to None.

    Raises
    ------
    ValueError
        If both `name` and `description` are None and an update is attempted.

    Examples
    --------
    >>> update_info = TuneUpdateIn(name='Updated Tune Job', description='Updated description')
    >>> update_info.name
    'Updated Tune Job'
    """

    name: Optional[str] = None
    description: Optional[str] = None
    train_options: Optional[dict] = {}


class TuneUpdateOut(TuneUpdateIn): ...


class TuneAndInferenceModel(BaseModel):
    """
    Combined response model for tunes and inference models.

    Attributes
    ----------
    id : str
        A unique identifier for tune or inference model.
    name : str
        The name of the tune or inference model.
    status : str, optional
        The current status of the tune or inference model. Defaults to an empty string.
    shared: bool
        Whether the tune/model is sharable accross users. Defaults to False.
    description : str
        A description of the tune or inference model.
    model_type : str, optional
        Whether it is a tune or inference model. Defaults to an empty string.

    Raises
    ------
    ValueError
        If any required fields are not provided or if they contain invalid values.

    Examples
    --------
    >>> tunes_model_response = TuneAndInferenceModel(id='123', name='Tuning Job 1', description='Description of the job')
    >>> tunes_model_response.status
    ''
    """

    id: str
    name: str
    status: Optional[str] = ""
    shared: Optional[bool] = False
    description: str
    model_type: Optional[str] = ""
    active_: bool
    created_by_: Optional[str] = ""
    created_at_: Optional[datetime] = None
    updated_at_: Optional[datetime] = None

    @model_validator(mode="after")
    def update_shared(self):
        self.shared = self.created_by_ == settings.DEFAULT_SYSTEM_USER
        return self

    class Config:
        from_attributes = True


class TunesAndInferenceModels(ListResponse):
    """Response Model for Tunes And Inference Models list

    Attributes
    ----------
    ListResponse : list of TuneAndInferenceModel
        List of tunes and inference model

    Examples
    --------
    >>> tunes_model_response = TunesAndInferenceModels(results=[TuneAndInferenceModel(id='1', name='Tune and Inference Model Job 1', description='First tuning job')])
    >>> tunes_model_response.results[0].name
    'Tune and Inference Model Job 1'
    """

    results: list[TuneAndInferenceModel]


class TunedModelDeployIn(BaseModel):
    """Model for Deploying Tune

    Attributes
    ----------
    model_name : str
        Name of the model being onboarded.
    description : str
        The description of the model being onboarded.
    model_style_id: str, Optional
        The dataset to be used for the tuning job. Defaults to None.
    sharable: bool
        Whether the model is sharable accross users. Defaults to False.


    Raises
    ------
    ValueError
        If any required fields are not provided or if they contain invalid values.

    Examples
    --------
    >>> tuned_model_deploy = TunedModelDeployIn(model_name= "some_name", description = "some description")
    >>> tuned_model_deploy.model_name
    'some_name'
    """

    model_name: str = Field(
        description="Name of the model being onboarded.",
        min_length=4,
        max_length=40,
    )
    model_description: str = Field(
        description="Brief description of the fine-tuned model being onboarded."
    )
    model_style_id: Optional[str] = None
    data_type: Optional[str] = "s2"
    sharable: Optional[bool] = False
    collections: Optional[dict] = None
    scaled_bands: Optional[list] = []
    scaling_factor: Optional[float] = 1.0
    resolution: Optional[float] = 10.0
    extra_info: Optional[dict] = None

    class Config:
        protected_namespaces = ()

    @field_validator("model_name")
    @classmethod
    def validate_name(cls, model_name: str) -> str:
        """Validates the tune model_name

        Parameters
        ----------
        model_name : str
            The name of the tune.

        Returns
        -------
        str
            Cleaned up name of the tune without special characters or white spaces.

        Raises
        ------
        ValueError
             If `name` contains special characters or white spaces.
        """
        # Clean-up the tune name.
        model_name = model_name.replace(" ", "-").replace("_", "-").strip()
        if not re.match("^[a-zA-Z0-9]+([.-]{0,1}[a-zA-Z0-9]+)*$", model_name):
            raise ValueError(
                "must not contain special characters or white spaces. Replace underscores with hyphens."
            )
        return model_name


class MlflowRunMetrics(BaseModel):
    """
    Model for storing metrics of a single run in MLflow.

    Attributes
    ----------
    name : str
        A name for the run derived from info.run_name.
    status : {'FINISHED', 'NOT_FOUND', 'ERROR', 'RUNNING'}
        The current status of the run, represented as a literal type.
        'FINISHED' - Run has completed succesfully
        'NOT_FOUND' - Run not created in the database
        'ERROR' - Run encountered error while runnning
        'RUNNING' - Run is currently running
    epochs : str
        The number of epochs the model has been trained.
    metrics : list of dict
        A list of metrics associated with the current run. Each dictionary contains metric details.

    Raises
    ------
    ValueError
        If the `status` is not one of the expected values ('FINISHED', 'NOT_FOUND', 'ERROR', 'RUNNING')
          or if required attributes are missing.

    Examples
    --------
    >>> run_metrics = MlflowRunMetrics(
    ...     name='Test',
    ...     status='FINISHED',
    ...     epochs='1',
    ...     metrics=[{"accuracy": 0.95, "loss": 0.05}],
    ... )
    >>> run_metrics.status
    'FINISHED'
    """

    name: str
    status: Literal["FINISHED", "NOT_FOUND", "ERROR", "RUNNING"]
    epochs: str
    metrics: list[dict]


class TunedModelMlflowMetrics(BaseModel):
    """
    Model for storing metrics of tuned models in MLflow.

    Attributes
    ----------
    id : str
        A unique identifier for the tuned model.
    status : {'FINISHED', 'NOT_FOUND', 'ERROR', 'RUNNING'}
        The current status of the tuned model, represented as a literal type.
        'FINISHED' - Tune has completed succesfully
        'NOT_FOUND' - Tune not created in the database
        'ERROR' - Tune encountered error while runnning
        'RUNNING' - Tune is currently running
    runs: list of dict
        A list of MlflowRunMetrics associated with the current tune.
    details : str
        Additional details about the tuned model.

    Raises
    ------
    ValueError
        If the `status` is not one of the expected values ('FINISHED', 'NOT_FOUND', 'ERROR', 'RUNNING')
          or if required attributes are missing.

    Examples
    --------
    >>> model_metrics = TunedModelMlflowMetrics(
    ...     id='model_123',
    ...     status='FINISHED',
    ...     runs=MlflowRunMetrics(
    ...         name='Test',
    ...         status='FINISHED',
    ...         epochs='1',
    ...         metrics=[{"accuracy": 0.95, "loss": 0.05}],
    ...     )
    ...     details='Trained on dataset XYZ'
    ... )
    >>> model_metrics.status
    'FINISHED'
    """

    id: str
    status: Literal["FINISHED", "NOT_FOUND", "ERROR", "RUNNING"]
    runs: list[MlflowRunMetrics]
    details: str


class TuneDownloadOut(BaseModel):
    """
    Model for downloading Tune resources.

    Attributes
    ----------
    id : str
        A unique identifier for the Tune.
    name : str
        The name of the Tune.
    description : str
        A brief description of the Tune.
    config_url : str
        Presigned COS URL to download the configuration file associated with the Tune.
    checkpoint_url : str
        Presigned COS URL to download the checkpoint file associated with the Tune.

    Raises
    ------
    ValueError
        If any of the URLs are not valid or if required attributes are missing.

    Examples
    --------
    >>> tune_download = TuneDownloadOut(
    ...     id='tune_001',
    ...     name='Tune 1',
    ...     description='Downloadable resources for Tune 1',
    ...     config_url='https://s3.com/config/tune_001.json',
    ...     checkpoint_url='https://s3.com/checkpoints/tune_001.ckpt'
    ... )
    >>> tune_download.name
    'Tuning Job 1'
    """

    id: str
    name: str
    description: str
    config_url: str
    checkpoint_url: str


class TuneSubmitOut(BaseModel):
    """
    Model for the output of a tuning job submission.

    Attributes
    ----------
    tune_id : str
        A unique identifier for the submitted tuning job.
    mcad_id : str, optional
        Identifier for the MCAD. Defaults to None.
    status : {'Pending', 'Submitted', 'In_progress', 'Failed', 'Finished', 'Error'}
        The current status of the tuning job submission, represented as a literal type.
        'Pending' - Tune is yet to be submitted to the Kubernetes job
        'Submitted' - Tune has been submitted to the Kubernetes job
        'In_progress' -  Tune is currently running
        'Failed' - Tune encountered an error while running
        'Error' - Tune never started running due to some error
    message : dict, optional
        Additional information or error messages associated with the submission. Defaults to None.

    Raises
    ------
    ValueError
        If the `status` is not one of the expected values
         ('Pending', 'Submitted', 'In_progress', 'Failed', 'Finished', 'Error').

    Examples
    --------
    >>> tune_submission = TuneSubmitOut(
    ...     tune_id='tune_001',
    ...     mcad_id='mcad_001',
    ...     status='Pending',
    ...     message={"info": "Submission is being processed."}
    ... )
    >>> tune_submission.status
    'Pending'
    """

    tune_id: str
    mcad_id: Optional[str] = None
    status: Literal[
        "Pending", "Submitted", "In_progress", "Failed", "Finished", "Error"
    ]
    message: Optional[dict] = None


# ***************************************************
# Tasks Schemas
# ***************************************************
class TaskPurposeEnum(str, enum.Enum):
    REGRESSION = "Regression"
    SEGMENTATION = "Segmentation"
    OTHER = "Other"
    MULTIMODAL = "Multimodal"

    def __str__(self) -> str:
        """Generates string representation of the ModelCategory

        Returns
        -------
        str
            String representation of ModelCategory value
        """
        return self.value

    @classmethod
    def _missing_(cls, value: str):
        # for case insensitive input mapping
        return cls.__members__.get(value.upper(), None)


class TaskIn(BaseModel):
    """
    Model for input parameters to create a new task.

    Attributes
    ----------
    name : str
        The name of the task. This field is required.
    description : str, optional
        A brief description of the task. Defaults to None.
    content : str
        A Base64 encoded string representing a fine-tuning YAML template.
    model_params : Any, optional
        Parameters for the model associated with the task. Defaults to an empty dictionary.
    extra_info : dict, optional
        Additional parameters for the task, such as runtime image information used to run the Tune task.
        Defaults to {'runtime_image': ''}.
    dataset_id: str, optional
        Dataset ID for the created task

    Raises
    ------
    ValueError
        If the `content` field is not a valid Base64 encoded string or if the `name` field is empty.

    Examples
    --------
    >>> task_input = TaskIn(
    ...     name='My Fine-tuning Task',
    ...     description='Task for fine-tuning a model.',
    ...     content='c29tZSBCYXNlNjQgc3RyaW5n',
    ...     model_params={"param1": "value1"},
    ...     extra_info={"runtime_image": "us.icr.io/gfmaas/geostudio-ft-deploy:v3"}
    ... )
    >>> task_input.name
    'My Fine-tuning Task'
    """

    name: str
    description: Optional[str] = None
    purpose: Optional[TaskPurposeEnum] = Field(
        description="The use case for this task",
        default=TaskPurposeEnum.SEGMENTATION,
    )
    content: str = Field(
        description="Base64 encoded string of a fine-tuning yaml template."
    )
    model_params: Optional[Any] = {}
    extra_info: Optional[dict] = Field(
        description="Extra params e.g {'runtime_image': 'us.icr.io/gfmaas/geostudio-ft-deploy:v3'}",
        default={"runtime_image": ""},
    )
    dataset_id: Optional[str] = None

    class Config:
        protected_namespaces = ()


class TaskOut(ItemResponse):
    """
    Response model for a tuning task.

    Attributes
    ----------
    name : str
        The name of the tuning task.
    description : str
        A description of the tuning job.
    model_params : dict or str, optional
        Parameters for the model associated with the tuning task. Defaults to an empty dictionary.

    Raises
    ------
    ValueError
        If any required fields are not provided or if they contain invalid values.

    Examples
    --------
    >>> task_out = TaskOut(name="some_name", description="some description")
    >>> task_out.name
    "some_name"
    """

    name: str
    description: str
    purpose: Optional[str] = None
    model_params: Optional[Union[dict, str]] = {}
    extra_info: Optional[dict] = None

    class Config:
        protected_namespaces = ()


class TasksOutRecord(ItemResponse):
    """
    Response model for a tuning task record.

    Attributes
    ----------
    name : str
        The name of the tuning task record.
    description : str
        A description of the tuning job.
    Raises
    ------
    ValueError
        If any required fields are not provided or if they contain invalid values.

    Examples
    --------
    >>> task_out_record = TasksOutRecord(name="some_name", description="some description")
    >>> task_out_record.name
    "some_name"
    """

    name: str
    description: str
    purpose: Optional[str] = None
    extra_info: Optional[dict] = None

    class Config:
        protected_namespaces = ()


class TasksOut(ListResponse):
    """Response Model for Tasks list

    Attributes
    ----------
    ListResponse : list of TasksOutRecord
        List of tuning tasks

    Examples
    --------
    >>> tune_tasks_response = TasksOut(results=[TasksOutRecord(name='Task 1', description='First tuning task')])
    >>> tune_tasks_response.results[0].name
    'Task 1'
    """

    results: list[TasksOutRecord]


class TryOutTuneInput(BaseModel):
    model_display_name: str = ""
    # model_id: Optional[UUID] = None
    description: Optional[str] = "try-out"
    location: str
    geoserver_layers: Optional[Dict[str, Any]] = None
    spatial_domain: SpatialDomain
    temporal_domain: List[str]
    model_input_data_spec: Optional[List[Dict[str, Any]]] = None
    data_connector_config: Optional[List[DataSource]] = None
    geoserver_push: Optional[List[GeoServerPush]] = None
    post_processing: Optional[Dict[str, Any]] = None
    maxcc: Optional[int] = 100
    model_config = {"extra": "allow"}


class UploadTuneInput(BaseModel):
    name: str
    description: str
    tune_config_url: str
    tune_checkpoint_url: str
    model_input_data_spec: Optional[List[Dict[str, Any]]] = None
    data_connector_config: Optional[List[DataSource]] = None
    geoserver_push: Optional[List[GeoServerPush]] = None
    post_processing: Optional[Dict[str, Any]] = None


class TryOutSubmitInference(TryOutTuneInput):
    fine_tuning_id: Optional[str] = None


class TryOutResponse(BaseModel):
    inference_id: uuid.UUID
    description: Optional[str] = None
    location: str
    status: str
    model_id: uuid.UUID
    created_at: Optional[datetime] = None

    @model_validator(mode="before")
    def update_count(cls, values):
        if isinstance(values, dict):
            values["inference_id"] = values["id"]
        else:
            values.inference_id = values.id
        return values


# ***************************************************
# Dataset Schemas
# ***************************************************
class DatasetOut(ItemResponse):
    """
    Response model for a dataset.

    Attributes
    ----------
    id : str
        The id of the dataset.
    name : str
        The name of the dataset.
    description : str
        A description of the dataset.
    status : str
        The status of the dataset.
    data_mapping : object, optional
        Values to map to the dataset factory schema. Defaults to an empty dictionary.
    model_params : dict or str, optional
        Parameters for the model associated with the dataset. Defaults to an empty dictionary.

    Raises
    ------
    ValueError
        If any required fields are not provided or if they contain invalid values.

    Examples
    --------
    >>> dataset_out = DatasetOut(id='1', name="some_name", description="some description",status="Pending")
    >>> dataset_out.status
    "Pending"
    """

    id: str
    name: str
    description: str
    status: str
    data_mapping: Optional[object] = None
    model_params: Optional[Union[dict, str]] = {}

    class Config:
        protected_namespaces = ()

    # @validator("model_params", pre=True, always=True)
    # def update_model_params(cls, val):
    #     if isinstance(val, str):
    #         try:
    #             val = json.loads(val)
    #         except json.decoder.JSONDecodeError:
    #             val = {}
    #     return val


class DatasetsOut(ListResponse):
    """Response Model for Datasets list

    Attributes
    ----------
    ListResponse : list of DatasetOut
        List of Datasets

    Examples
    --------
    >>> datasets_response = DatasetsOut(results=[
    >>>     DatasetOut(id='1', name="some_name", description="some description",status="Pending")])
    >>> datasets_response.results[0].name
    'some_name'
    """

    results: list[DatasetOut]

    class Config:
        protected_namespaces = ()


class SampleDatasetsOut(BaseModel):
    """
    Model for the output of sample datasets.

    Attributes
    ----------
    id : str
        A unique identifier for the sample dataset.
    status : str
        The current status of the sample dataset.
    sample_images : list, optional
        A list of sample images associated with the dataset. Defaults to an empty list.

    Raises
    ------
    ValueError
        If the `id` or `status` fields are empty.

    Examples
    --------
    >>> sample_dataset = SampleDatasetsOut(
    ...     id='sample_001',
    ...     status='Finished',
    ...     sample_images=['image1.tif', 'image2.tif']
    ... )
    >>> sample_dataset.status
    'Finished'
    """

    id: str
    status: str
    sample_images: Optional[list] = []


class DatasetIn(BaseModel):
    """
    Model for input parameters to create a new dataset.

    Attributes
    ----------
    dataset_name : str
        The name of the dataset. This field is required.
    label_suffix : str
        The suffix for the labels
    dataset_url : str
        The url to a zip folder with the data
    custom_bands : list of dict, optional
        A list of custom bands for the task, where each band is represented by a dictionary
        containing an 'id' and 'value'. Defaults to a predefined set of bands:
        [{"id": 0, "value": "Red"}, {"id": 1, "value": "Blue"}, {"id": 2, "value": "Green"}].
    description : str, optional
        A brief description of the dataset. Defaults to None.
    training_data_suffix: str
        The suffix for the training images
    purpose: {"Regression", "Segmentation", "Generate", "NER", "Classify", "Other"}
        Purpose of the dataset.
    label_categories : list, optional
        Categories of the labels provided. Defaults to empty list.
    training_params : dict, optional
        Additional training parameters for the dataset. Defaults to empty dictionary.

    Raises
    ------
    ValueError
    ValueError
        If the `purpose` is not one of the expected values
          ("Regression", "Segmentation", "Generate", "NER", "Classify", "Other")
          or if required attributes are missing.

    Examples
    --------
    >>> dataset_input = DatasetIn(
    ...     dataset_name='dataset-one',
    ...     label_suffix="_label.tif",
    ...     dataset_url="https://box.com/file.zip"
    ...     training_data_suffix="_image.tif"
    ...     purpose='Regression'
    ... )
    >>> dataset_input.purpose
    'Regression'
    """

    dataset_name: str
    label_suffix: str
    dataset_url: str
    custom_bands: Optional[List[dict]] = []
    description: Optional[str]
    training_data_suffix: str
    purpose: Literal[
        "Regression", "Segmentation", "Generate", "NER", "Classify", "Other"
    ]
    label_categories: Optional[List[dict]] = []
    training_params: Optional[Dict] = {}

    class Config:
        protected_namespaces = ()
        load_instance = True


# ***************************************************
# BaseModels Schemas
# ***************************************************
class ModelCategory(str, enum.Enum):
    terramind = "terramind"
    prithvi = "prithvi"
    clay = "clay"
    dofa = "dofa"
    resnet = "resnet"
    convnext = "convnext"

    def __str__(self) -> str:
        """Generates string representation of the ModelCategory

        Returns
        -------
        str
            String representation of ModelCategory value
        """
        return self.value


class BaseModelParamsIn(BaseModel):
    backbone: Optional[str] = Field(description="the base model backbone", default="")
    patch_size: Optional[int] = Field(description="num_layers", default=16)
    num_layers: Optional[int] = Field(description="num_layers", default=12)
    embed_dim: Optional[int] = Field(description="embed_dim", default=768)
    num_heads: Optional[int] = Field(description="num_heads", default=12)
    tile_size: Optional[int] = Field(description="tile_size", default=1)
    tubelet_size: Optional[int] = Field(description="tubelet_size", default=1)
    model_category: Optional[ModelCategory] = Field(
        description="model_category", default="prithvi"
    )


class BaseModelsIn(BaseModel):
    """
    Model for input parameters to create a new Base Model.

    Attributes
    ----------
    name : str
        The name of the Base Model. This field is required.
    description : str, optional
        A brief description of the Base Model. Defaults to None.
    checkpoint_filename : str
        The filename of the  checkpoint for the base model. Defaults to an empty string.


    Raises
    ------
    ValueError
        If the `name` and `description` fields are empty.

    Examples
    --------
    >>> base_model_input = BaseModelsIn(
    ...     name='Prithvi Base Model',
    ...     description='Base Model ',
    ... )
    >>> base_model_input.name
    'Prithvi Base Model'
    """

    name: str
    description: str
    checkpoint_filename: Optional[str] = ""
    model_params: Optional[BaseModelParamsIn] = BaseModelParamsIn()

    class Config:
        protected_namespaces = ()


class BaseModelOut(ItemResponse):
    """
    Response model for a Base Model.

    Attributes
    ----------
    name : str
        The name of the Base Model.
    description : str
        A description of the Base Model.
    checkpoint_filename : str, optional
        The filename of the  checkpoint for the base model.
    model_params : dict or ModelBaseParams, optional
        Parameters for the Base Model. Defaults to an empty dictionary and is validated before use.

    Raises
    ------
    ValueError
        If any required fields are not provided or if they contain invalid values.
        If the `model_params` cannot be parsed into a valid JSON object.

    Examples
    --------
    >>> base_model_out = BaseModelOut(name="Prithvi Base Model", description="some description")
    >>> base_model_out.name
    "Prithvi Base Model"
    """

    name: str
    description: str
    checkpoint_filename: Optional[str]
    model_params: Optional[Union[dict, ModelBaseParams]] = {}
    status: Optional[str] = "Finished"
    shared: Optional[bool] = False

    class Config:
        protected_namespaces = ()

    @field_validator("model_params", mode="before")
    def update_model_params(cls, val):
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except json.decoder.JSONDecodeError:
                val = {}
        return val

    @model_validator(mode="before")
    def update_shared(cls, values):
        if isinstance(values, dict):
            values["shared"] = settings.DEFAULT_SYSTEM_USER == values["created_by"]
        else:
            values.shared = settings.DEFAULT_SYSTEM_USER == values.created_by
        return values


class BaseModelParamsOut(BaseModel):
    """
    Model for the output parameters of a base model.

    Attributes
    ----------
    model_params : dict or ModelBaseParams, optional
        Parameters for the model. Can be a dictionary or an instance of `ModelBaseParams`.
        Defaults to an empty dictionary.

    Methods
    -------
    update_model_params(val: Any) -> Union[dict, ModelBaseParams]
        Validates and updates the `model_params` attribute. If the input is a string, it attempts
        to parse it as JSON. If parsing fails, it defaults to an empty dictionary.

    Raises
    ------
    ValueError
        If the `model_params` input is not a valid dictionary or JSON string.

    Examples
    --------
    >>> base_model_params = BaseModelParamsOut(
    ...     model_params={"param1": "value1", "param2": "value2"}
    ... )
    >>> base_model_params.model_params
    {'param1': 'value1', 'param2': 'value2'}

    """

    model_params: Optional[Union[dict, BaseModelParamsIn]] = {}

    class Config:
        protected_namespaces = ()

    @field_validator("model_params", mode="before")
    def update_model_params(cls, val):
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except json.decoder.JSONDecodeError:
                val = {}
        return val


class BaseModelsOut(ListResponse):
    """Response Model for BaseModel list

    Attributes
    ----------
    ListResponse : list of BaseModelsOut
        List of BaseModels

    Examples
    --------
    >>> base_model_response = BaseModelOut(results=[BaseModelsOut(
    >>>     name="Prithvi Base Model", description="some description"])
    >>> base_model_response.results[0].name
    'Prithvi Base Model'
    """

    results: list[BaseModelOut]


class FilesShareOut(BaseModel):
    """Response Model for file-share get endpoint."""

    upload_url: str
    download_url: str
    message: str
