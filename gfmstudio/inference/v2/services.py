# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import json
import random
import uuid
from ast import literal_eval
from typing import Union

from fastapi import HTTPException
from redis.asyncio.client import Redis
from requests import Session

from gfmstudio.common.api import utils
from gfmstudio.common.api.crud import ItemCrud, ModelType
from gfmstudio.config import settings
from gfmstudio.data_advisor.data_advisor import invoke_data_advisor_service
from gfmstudio.inference.errors import errors
from gfmstudio.inference.integration_adaptors.gateway_pipelines import (
    invoke_pipelines_orchestration_service,
)
from gfmstudio.inference.services import notify_inference_webhook_events
from gfmstudio.inference.types import EventStatus
from gfmstudio.inference.v2.models import Inference, Model
from gfmstudio.inference.v2.schemas import (
    InferenceGetResponseWebhooks,
    V2PipelineCreate,
)
from gfmstudio.log import logger


def merge_bounding_boxes(bbox1, bbox2):
    """
    Merge two bounding boxes to create a larger bounding box that encompasses both.

    Args:
        bbox1: List of [min_x, min_y, max_x, max_y] for first bounding box
        bbox2: List of [min_x, min_y, max_x, max_y] for second bounding box

    Returns:
        List of [min_x, min_y, max_x, max_y] for the merged bounding box
    """
    # Extract coordinates from both bounding boxes
    min_x1, min_y1, max_x1, max_y1 = bbox1
    min_x2, min_y2, max_x2, max_y2 = bbox2

    # Find the minimum and maximum coordinates across both boxes
    merged_min_x = min(min_x1, min_x2)
    merged_min_y = min(min_y1, min_y2)
    merged_max_x = max(max_x1, max_x2)
    merged_max_y = max(max_y1, max_y2)

    return [merged_min_x, merged_min_y, merged_max_x, merged_max_y]


async def update_inference_status(
    inference_id: str,
    status: str,
    db: Session,
    user: str,
    event_error: str = None,
):
    """Update the status of an inference task.

    Parameters
    ----------
    inference_id : str
        The inference id.
    status : str
        Status of the inference task.
    event_error : str, optional
        The error arising from an event, by default None
    db : Session, optional
        db session, by default None

    """
    logger.info("Updating the status for Inference-%s, user-%s", inference_id, user)
    inference_crud = ItemCrud(model=Inference)
    update_item = {"status": status}
    if event_error:
        update_item["inference_output"] = {"errors": [{"message": event_error}]}

    updated_item = inference_crud.update(
        db=db,
        item_id=inference_id,
        item=update_item,
        user=user,
    )
    return updated_item


async def invoke_inference_v2_pipelines_handler(
    payload: Union[ModelType, dict],
    user: str,
    notify: bool = False,
    session: Session = None,
    **kwargs,
):
    """Invoke Inference pipelines service.

    Parameters
    ----------
    payload : Union[ModelType, dict]
        Payload sent to the inference service
    notify : bool, optional
        Toggle to push notifications to redis, by default False

    Returns
    -------
    dict
        Json response data from the inference pipelines service.

    Raises
    ------
    HTTPException
        Missing data or service unavailable.

    """
    session = session or next(utils.get_db())
    inference_id = payload["inference_id"]
    model_name = payload.get("model_internal_name")

    logger.info(f"\n*********** {model_name}: ***********")
    logger.info(f"INF> {inference_id}: Starting inference v2 pipeline.")

    payload["model_id"] = model_name
    service_payload = V2PipelineCreate(**payload).model_dump()
    logger.info(f"Inference Data = {service_payload}")

    # check data availability
    model_input_data_spec = service_payload["model_input_data_spec"]
    connector = model_input_data_spec[0]["connector"]
    collection = model_input_data_spec[0]["collection"]
    bbox = service_payload["spatial_domain"]["bbox"]
    maxcc = service_payload["maxcc"]

    if settings.DATA_ADVISOR_ENABLED and len(bbox[0]) > 0:
        try:
            event = payload["inference_id"]
            logger.info(
                f"{event} - Data Advisor Enabled... Taking a peek at available data 👀🔎🔎..."
            )

            dates = service_payload["temporal_domain"]
            params = {
                "data_connector": connector,
                "collections": [collection],
                "dates": dates,
                "bbox": bbox,
                "maxcc": maxcc,
            }
            logger.debug(f"Data advisor params: {params}")
            resp = await invoke_data_advisor_service(**params)
        except Exception as e:
            logger.error(f"Data advisor error: {e}")
        else:
            # Create a db session for async inference runs
            # if resp.status_code // 100 == 2:
            if resp:
                # err_detail = resp.json().get("message") or errors["1001"]["uiMessage"]
                err_detail = errors["1001"]["uiMessage"]
                count = len(bbox)
                for i, j in enumerate(resp["results"]):
                    error_detail = j.get("message")
                    if error_detail:
                        bounding_box = resp["results"][i]["bbox"]
                        logger.info(
                            f"Data Advisor data unavailable for bbox:  {bounding_box}..."
                            f"{error_detail}"
                        )
                        service_payload["spatial_domain"]["bbox"].remove(bounding_box)
                        count -= 1
                if count == 0:
                    # All bounding boxes or urls have no data... Update status of the inference.
                    logger.info(
                        f"{event} - Data Advisor data unavailable ... Inferencing Failed ..."
                    )
                    await update_inference_status(
                        inference_id=inference_id,
                        status=EventStatus.FAILED,
                        event_error=err_detail,
                        user=user,
                        db=session,
                    )
                    return await notify_inference_webhook_events(
                        channel=kwargs.get("channel"),
                        message={"error": err_detail},
                        detail_type=kwargs.get("detail_type"),
                        status=EventStatus.FAILED,
                    )
                else:
                    res = resp["results"]
                    logger.debug(f"Data advisor results: {res}")
                    logger.info(
                        f"{event} - Data Advisor data available ... Inferencing ..."
                    )
            else:
                logger.warning(
                    f"{event} - Data-advisor service Error (status-code: {resp.status_code})"
                    f"...Running inference without data-availability check. \n Error {resp.json()}",
                )
    try:
        inference_run_resp = invoke_pipelines_orchestration_service(
            inference_id=inference_id,
            payload=service_payload,
            deploy_model=kwargs.get("deploy_model", False),
            pipeline_version=kwargs.get("pipeline_version"),
        )
    except Exception as exc:
        logger.exception("Error when running inference.")
        # Incoming details format is "error_code: error message"
        try:
            code = exc.details().split(":")[0]
            error_info = errors.get(code, "n/a")
        except:  # noqa: E722
            error_info = "n/a"

        if "Connection aborted" in str(exc):
            code = "1017"
            error_info = errors.get(code, "n/a")

        try:
            if error_info == "n/a":
                # show validation errors
                if '"code":"RequestValidationError"' in exc.details():
                    detail = literal_eval(json.loads(exc.details())["description"])[0][
                        "msg"
                    ]
                else:
                    detail = f"Inference service for {payload['model_id']} temporarily unavailable."
            else:
                detail = error_info.get(
                    "uiMessage",
                    f"Inference service for {payload['model_id']} temporarily unavailable with unkown error.",
                )
        except:  # noqa: E722
            detail = "An unknown error occured when inferencing."

        await update_inference_status(
            inference_id=inference_id,
            status=EventStatus.FAILED,
            event_error=str(detail),
            user=user,
            db=session,
        )
        raise

    if inference_run_resp.status_code // 100 != 2:
        await update_inference_status(
            inference_id=inference_id,
            status=EventStatus.FAILED,
            event_error=f"Inference Pipeline Error: {str(inference_run_resp.text)}",
            user=user,
            db=session,
        )


async def save_predicted_layers(
    db: Session,
    layer_group_id: str,
    layers: list = None,
    bbox_pred: list = None,
    user: str = None,
    output_url: str = None,
):
    """Saves inference predicted layers to Layer model and updates inference status.

    Parameters
    ----------
    layer_group_id : str
        The primary key id of the inference run.
    layers : list
        A list of predicted layers.
    bbox_pred : list, optional
        A list of cordinates respresenting the bounding boxes
        of the predicted region, by default None
    user : str, optional
        The logged in user, by default None
    output_url : str, optional
        Output url of predicted layers, by default None
    db : Session, optional
        db session, by default None

    """
    inference_crud = ItemCrud(model=Inference)

    # Update the LayerGroup with layers.
    user = user or settings.DEFAULT_SYSTEM_USER
    if layers:
        item = inference_crud.get_by_id(db, layer_group_id, user=user)
        if not item:
            raise HTTPException(status_code=404, detail="Inference not found")

        geoserver_layers = item.geoserver_layers or {}
        predicted_layers = geoserver_layers.get("predicted_layers", [])
        existing_bboxes = geoserver_layers.get("bbox_pred")
        existing_layer_urls = {
            item.get("uri") for item in predicted_layers if "uri" in item
        }
        for layer in layers:
            coverage_store = layer.get("coverageStore", {})
            workspace_name = coverage_store.get("workspace", {}).get("name")
            layer_name = coverage_store.get("name", "")
            uri = f"{workspace_name}:{layer_name}"
            display_name = (
                layer.get("display_name", "") or layer_name.rsplit("-", 1)[-1]
            )

            if uri not in existing_layer_urls:
                predicted_layers.append(
                    {
                        "uri": uri,
                        "display_name": display_name,
                        "sld_body": layer.get("layer_style_xml", ""),
                        "z_index": layer.get("z_index", random.randint(10, 100)),
                        **(
                            {"visible_by_default": layer.get("visible_by_default")}
                            if layer.get("visible_by_default")
                            else {}
                        ),
                    }
                )

        if existing_bboxes and bbox_pred[0] not in existing_bboxes:
            existing_bboxes.append(bbox_pred[0])
            merged_bboxes = merge_bounding_boxes(
                bbox1=bbox_pred[0], bbox2=existing_bboxes[0]
            )
            if merged_bboxes not in existing_bboxes:
                existing_bboxes.append(merged_bboxes)
        bbox_pred = existing_bboxes if existing_bboxes else bbox_pred

        # TODO: Add support for updating existing layers.
        geoserver_layers["predicted_layers"] = predicted_layers
        geoserver_layers["bbox_pred"] = bbox_pred
        update_fields = {
            "geoserver_layers": geoserver_layers,
        }
        if output_url:
            update_fields["inference_output"] = {"output_url": output_url}
        updated_item = inference_crud.update(
            db=db,
            item_id=layer_group_id,
            item=update_fields,
            user=user,
        )
    elif output_url:
        updated_item = inference_crud.update(
            db=db,
            item_id=layer_group_id,
            item={
                "geoserver_layers": {"predicted_layers": []},
                "inference_output": {"output_url": output_url},
            },
            user=user,
        )
    else:
        updated_item = None
    return updated_item


async def cleanup_autodeployed_model_resources(
    db, user, model_id: str, model_name: str
):
    model_crud = ItemCrud(model=Model)
    model_crud.update(
        db=db,
        item_id=str(model_id),
        item={
            "deleted": True,
            "active": False,
            "name": f"{model_name}-{str(uuid.uuid4())[:8]}",
        },
        protected=False,
    )


async def update_inference_webhook_events(
    *,
    user: str,
    inference_id: uuid.UUID,
    db: Session = None,
    event_detail: dict = None,
):
    """Updates inference entry with response from a webhook event."""
    try:
        predicted_layers = event_detail.get("predicted_layers", [])
        pred_bboxes = event_detail.get("bboxes", [])
        output_url = event_detail.get("output_url", None)
    except Exception:
        predicted_layers = []

    update_fields = {"layer_group_id": inference_id}
    if predicted_layers:
        update_fields["layers"] = predicted_layers
    if pred_bboxes:
        update_fields["bbox_pred"] = pred_bboxes
    if output_url:
        update_fields["output_url"] = output_url

    await save_predicted_layers(
        db=db,
        user=user,
        **update_fields,
    )
    logger.info("Updated inference-%s from webhooks", inference_id)


async def send_websocket_notification(
    updated_item,
    channel: str,
    detail_type: str,
    redis: Redis = None,
    status: str = None,
):
    # Notify listeners via Server Sent Events
    if updated_item:
        updated_dict = updated_item.__dict__
        updated_obj = InferenceGetResponseWebhooks(**updated_dict)
        await notify_inference_webhook_events(
            redis=redis,
            channel=channel,
            message=json.loads(updated_obj.model_dump_json()),
            detail_type=detail_type,
            status=status,
        )
        logger.debug("Notify notify_inference_webhook_events COMPLETE")
