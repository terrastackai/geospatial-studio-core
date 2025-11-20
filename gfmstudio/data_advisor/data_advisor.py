# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import backoff
import httpx
from fastapi import HTTPException
from terrakit import DataConnector

from gfmstudio.config import settings

from .helper_functions import find_data_bbox


@backoff.on_exception(
    backoff.expo, (httpx.HTTPStatusError, httpx.RequestError), max_tries=3
)
async def data_advisor_list_collections(data_connector: str):
    try:
        dc = DataConnector(connector_type=data_connector)
        collections = dc.connector.list_collections()

        return {"results": collections}
    except ValueError as e:
        error_message = str(e) if str(e) else "ValueError occurred"
        raise HTTPException(status_code=400, detail=error_message)


@backoff.on_exception(
    backoff.expo, (httpx.HTTPStatusError, httpx.RequestError), max_tries=3
)
async def invoke_data_advisor_service(
    *,
    data_connector: str = None,
    collections: list[str] = None,
    dates: list[str] = None,
    bbox: list[list[float]] = None,
    area_polygon: str = None,
    maxcc: float = settings.DATA_ADVISOR_MAX_CLOUD_COVER,
    urls: list[str] = None,
    model_id: str = None,
    pre_days: int = settings.DATA_ADVISOR_PRE_DAYS,
    post_days: int = settings.DATA_ADVISOR_POST_DAYS,
):
    dc = DataConnector(connector_type=data_connector)

    data = {
        "collections": collections,
        "dates": dates,
        "bbox": bbox,
        "area_polygon": area_polygon,
        "maxcc": maxcc,
        "pre_days": pre_days,
        "post_days": post_days,
    }
    response = find_data_bbox(connector=dc, payload=data)
    return response
