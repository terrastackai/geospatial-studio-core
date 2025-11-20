# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest
import requests

from gfmstudio.common.api import crud
from gfmstudio.fine_tuning import schemas
from gfmstudio.fine_tuning.dataset_schemas import GeoDatasetRequestSchemaV2
from gfmstudio.fine_tuning.models import BaseModels, GeoDataset, Tunes, TuneTemplate
from gfmstudio.inference.services import (
    backoff_hdlr,
    download_with_backoff,
    fatal_code,
    giveup_hdlr,
    invoke_cancel_inference_handler,
    invoke_tune_upload_handler,
)
from gfmstudio.inference.types import InferenceStatus
from gfmstudio.inference.v2.models import Inference, Model, Task
from gfmstudio.inference.v2.schemas import (
    InferenceConfig,
    InferenceCreate,
    SpatialDomain,
)

tune_template_crud = crud.ItemCrud(model=TuneTemplate)
bases_crud = crud.ItemCrud(model=BaseModels)
tunes_crud = crud.ItemCrud(model=Tunes)


model_crud = crud.ItemCrud(model=Model)
inference_crud = crud.ItemCrud(model=Inference)
task_crud = crud.ItemCrud(model=Task)


@patch("gfmstudio.inference.services.download_with_backoff")
@patch("gfmstudio.inference.services.boto3.client")
def test_boto_upload_called(mock_boto_client, mock_download, db):
    """
    Test that invoke_tune_upload_handler downloads config and checkpoint files,
    uploads them to S3, and updates tune status successfully.

    This test mocks HTTP downloads and S3 client uploads, and sets up necessary
    database records including dataset, base model, tune template, and tune instance.
    """
    # Mock two different HTTP responses for config and checkpoint downloads
    mock_response_config = MagicMock()
    mock_response_config.raw = MagicMock()

    mock_response_checkpoint = MagicMock()
    mock_response_checkpoint.raw = MagicMock()

    # Set side_effect so each call to download_with_backoff returns a different mock response
    mock_download.side_effect = [mock_response_config, mock_response_checkpoint]

    mock_s3_client = MagicMock()
    mock_boto_client.return_value = mock_s3_client

    user = "system@ibm.com"

    # Create dataset
    dataset_crud = crud.ItemCrud(model=GeoDataset)

    data = GeoDatasetRequestSchemaV2(
        **{
            "dataset_name": "sandbox-dataset",
            "label_suffix": ".mask.tif",
            "dataset_url": "https://example.com",
            "purpose": "Segmentation",
            "description": "sandbox dataset",
            "data_sources": [],
        }
    )

    created_dataset = dataset_crud.create(db=db, user=user, item=data)
    dataset_id = created_dataset.id if created_dataset else ""

    # create a base model
    data = schemas.BaseModelsIn(
        **{
            "name": "sandbox-base-model",
            "description": "base model",
        }
    )

    created_model = bases_crud.create(db=db, item=data, user=user)

    # create a tune template
    tune_template_data = TuneTemplate(
        **{
            "name": "my-test-name",
            "description": "my-test-description",
            "content": "",
            "task_schema": "",
            "model_params": {},
            "extra_info": {},
            "dataset_id": dataset_id,
            "purpose": schemas.TaskPurposeEnum.SEGMENTATION,
        }
    )
    tune_template = tune_template_crud.create(db=db, item=tune_template_data, user=user)

    # Create a Tunes model instance
    tune_instance = schemas.TuneSubmitIn(
        **{
            "name": "data_in.name",
            "description": "data_in.description",
            "status": "Pending",
            "dataset_id": dataset_id,
            "base_model_id": created_model.id,
            "tune_template_id": tune_template.id,
        }
    )

    created_tune = tunes_crud.create(
        db=db,
        item=tune_instance,
        user=user,
    )
    tune_id = created_tune.id

    result = asyncio.run(
        invoke_tune_upload_handler(
            tune_config_url="http://fake-url/config.yaml",
            tune_checkpoint_url="http://fake-url/checkpoint.ckpt",
            tune_id=tune_id,
            user=user,
            db=db,
        )
    )

    # Assert upload_fileobj was called twice (once for config, once for checkpoint)
    assert mock_s3_client.upload_fileobj.call_count == 2

    # Check returned result
    assert result == {"msg": "Upload complete"}


# test the exceptions
@pytest.mark.parametrize(
    "exception",
    [
        requests.exceptions.ConnectionError("connection failed"),
        requests.exceptions.Timeout("connection failed"),
    ],
)
@patch("gfmstudio.inference.services.download_with_backoff")
def test_invoke_tune_upload_raises_http_exception_on_network_error(
    mock_download, exception
):
    mock_download.side_effect = exception

    with pytest.raises(type(exception)) as exc_info:
        asyncio.run(
            invoke_tune_upload_handler(
                "http://fake-config-url",
                "http://fake-checkpoint-url",
                "test-tune-id",
                user="test-user",
                db=MagicMock(),
            )
        )

    assert "connection failed" in str(exc_info.value)


def test_fatal_code_with_4xx():
    """
    Test that fatal_code returns True for HTTP errors with 4xx status codes,
    indicating client errors are considered fatal.
    """
    mock_exception = requests.exceptions.HTTPError()
    mock_exception.response = MagicMock(status_code=404)
    assert fatal_code(mock_exception) is True


def test_fatal_code_with_5xx():
    """
    Test that fatal_code returns False for HTTP errors with 5xx status codes,
    since server errors are usually transient and not fatal.
    """
    mock_exception = requests.exceptions.HTTPError()
    mock_exception.response = MagicMock(status_code=503)
    assert fatal_code(mock_exception) is False


def test_fatal_code_with_no_response():
    """
    Test that fatal_code returns False when the exception has no response attribute,
    such as a ConnectionError, which should not be considered fatal based on HTTP status.
    """
    mock_exception = requests.exceptions.ConnectionError()
    assert fatal_code(mock_exception) is False


def test_backoff_hdlr_logs():
    """
    Verify backoff_hdlr logs the correct backoff message with retry details.
    """
    with patch("gfmstudio.log.logger.info") as mock_info:
        backoff_hdlr(
            {
                "wait": 1.0,
                "tries": 3,
                "target": backoff_hdlr,
                "args": ("http://example.com",),
                "kwargs": {},
            }
        )
        # Check the log message contains expected substring
        mock_info.assert_called()
        called_args = mock_info.call_args[0][0]  # first positional arg
        assert "Backing off 1.0s after 3 tries" in called_args


def test_giveup_hdlr_raises():
    """
    Verify giveup_hdlr raises ConnectionError with expected message after max retries.
    """
    with pytest.raises(requests.exceptions.ConnectionError) as exc_info:
        giveup_hdlr({"tries": 5, "args": ("http://example.com",)})
    assert "Max retries reached" in str(exc_info.value)


@patch("gfmstudio.inference.services.requests.get")
def test_download_with_backoff_success(mock_get):
    """
    Test that download_with_backoff successfully calls requests.get
    and returns a response with status code 200.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    response = download_with_backoff("http://example.com")
    assert response.status_code == 200
    mock_get.assert_called_once_with("http://example.com", stream=True, timeout=30)


# TEST  INFERENCE CANCELLATION LOGIC


@pytest.fixture
def user():
    return "system@ibm.com"


@pytest.fixture
def model(db, user):
    model_data = Model(
        internal_name="model-internal", display_name="Model Display", description="desc"
    )
    return model_crud.create(db=db, user=user, item=model_data)


@pytest.fixture
def inference(db, user, model):
    spatial_domain = SpatialDomain(
        bbox=[[-119.252472, 33.628342, -117.03650, 34.059309]],
        polygons=[],
        tiles=[],
        urls=[],
    )
    inference_config = InferenceConfig(
        spatial_domain=spatial_domain,
        temporal_domain=[
            "2024-12-18",
            "2025-01-14_2025-01-15",
            "2025-01-20",
        ],
    )
    inference_data = InferenceCreate(
        name="My Inference",
        description="some desc",
        model_id=model.id,
        inference_config=inference_config,
        spatial_domain=spatial_domain,
    )
    return inference_crud.create(db=db, user=user, item=inference_data)


def test_invoke_cancel_inference_handler(db, user, model, inference):
    inference_id = inference.id

    # Create tasks with various statuses
    # create a task with READY status
    task_crud.create(
        db=db,
        user=user,
        item=Task(
            inference_id=inference_id,
            status="READY",
            task_id=f"{uuid.uuid4()}-task_planning",
        ),
    )

    # create a task with PENDING status
    task_crud.create(
        db=db,
        user=user,
        item=Task(
            inference_id=inference_id,
            status="PENDING",
            task_id=f"{uuid.uuid4()}-task_planning",
        ),
    )

    asyncio.run(
        invoke_cancel_inference_handler(
            inference_id=inference_id,
            user=user,
            db_session=db,
        )
    )

    tasks = task_crud.get_all(db=db, user=user, filters={"inference_id": inference_id})
    task_statuses = [task.status for task in tasks]

    assert all(status in {"DONE", "FAILED", "STOPPED"} for status in task_statuses)
    assert "READY" not in task_statuses
    assert "PENDING" not in task_statuses

    updated_inference = inference_crud.get_by_id(db=db, item_id=inference_id, user=user)
    assert updated_inference.status == InferenceStatus.STOPPED


@patch("gfmstudio.inference.services.task_crud.get_by_id")
@patch("gfmstudio.inference.services.asyncio.sleep")
def test_invoke_cancel_inference_handler__waits_for_running_tasks(
    mock_sleep, mock_get_by_id, db, user, model, inference
):
    """
    Test that `invoke_cancel_inference_handler` polls and waits when
    tasks are in the RUNNING state, by checking that it repeatedly
    queries task status and calls asyncio.sleep for backoff before
    final cancellation when tasks transition to terminal states.
    """
    # Add a RUNNING task
    running_task = task_crud.create(
        db=db,
        user=user,
        item=Task(
            inference_id=inference.id,
            status="RUNNING",
            task_id=f"{uuid.uuid4()}-task_planning",
            created_by=user,
            updated_by=user,
        ),
    )

    task_id = running_task.id

    def side_effect(*args, **kwargs):
        task = task_crud.get_by_id(db=db, item_id=task_id, user=user)
        if mock_get_by_id.call_count == 1:
            task.status = "RUNNING"
        else:
            task.status = "FAILED"
        return task

    mock_get_by_id.side_effect = side_effect

    asyncio.run(
        invoke_cancel_inference_handler(
            inference_id=inference.id,
            user=user,
            db_session=db,
        )
    )

    assert mock_sleep.call_count >= 1
