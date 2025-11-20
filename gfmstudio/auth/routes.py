# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from requests import Session

from ..common.api import crud
from ..config import settings
from ..log import logger
from . import utils
from .api_key_utils import generate_apikey
from .authorizer import auth_handler, exchange_code_for_token, get_api_key
from .models import APIKey, User
from .schemas import (
    APIKeyListResponse,
    APIKeyRequestSchema,
    APIKeyResponseSchema,
    APIKeyUpdateSchema,
)

router = APIRouter(tags=["Studio / Authentication"])
apikey_crud = crud.ItemCrud(model=APIKey)
user_crud = crud.ItemCrud(model=User)


@router.get("/auth/token-uri", include_in_schema=False)
async def get_login_token_uri(request: Request):
    authorization_endpoint = settings.OAUTH_ENDPOINT
    client_id = settings.OAUTH_CLIENTID

    if authorization_endpoint and client_id:
        redirect_uri = f"{request.base_url}v1/auth/token".replace("//v1", "/v1")
        parse_oauth_url = urlparse(authorization_endpoint)
        base_oauth_url = f"{parse_oauth_url.scheme}://{parse_oauth_url.netloc}/v1.0/endpoint/default/authorize"  # noqa: E501
        login_url = f"{base_oauth_url}?approval_prompt=force&client_id={client_id}&redirect_uri={redirect_uri}&response_type=code&scope=openid+email+profile"  # noqa: E501
        logger.info("Redirecting to login url: %s", login_url)
        return {"AuthCodeUrl": login_url}

    return HTTPException(status_code=401)


@router.get("/auth/token", include_in_schema=False)
async def get_jwt_token(request: Request, code: str):
    access_token = request.headers.get("Authorization")
    if not access_token:
        redirect_uri = f"{request.base_url}v1/auth/token".replace("//v1", "/v1")
        access_token = exchange_code_for_token(code=code, redirect_uri=redirect_uri)
    return {"Authentication": access_token}


@router.get("/auth/api-keys", response_model=APIKeyListResponse)
async def list_apikeys(request: Request, auth=Depends(auth_handler)):
    """Gets an existing api-key or creates a new one if no API-Key exists."""
    email = auth[0]
    resp = await get_api_key(user_email=email)
    return {"results": resp or []}


@router.post("/auth/api-keys", response_model=APIKeyResponseSchema)
async def generate_apikey_token(
    request: Request,
    db: Session = Depends(utils.get_auth_db),
    auth=Depends(auth_handler),
):
    """Generate API key token."""
    email = auth[0]
    user = user_crud.get_all(db=db, ignore_user_check=True, filters={"email": email})
    if user_obj := user[0]:
        available_apikeys = [k for k in user_obj.apikeys if k.deleted is False]
        if len(available_apikeys) > 1:
            raise HTTPException(
                status_code=412,
                detail="You already have 2 API Keys registered. Delete one and try again.",
            )

        generated_api_key = generate_apikey()
        apikey_obj = APIKeyRequestSchema(
            value=generated_api_key["encrypted_key"],
            hashed_key=generated_api_key["hashed_key"],
            last_used_at=None,
            user_id=user_obj.id,
        )
        resp = apikey_crud.create(db=db, item=apikey_obj, user=email)
        return resp


@router.patch("/auth/api-keys", response_model=APIKeyResponseSchema)
async def apikey_activation(
    request: Request,
    apikey_id: uuid.UUID,
    item: APIKeyUpdateSchema,
    db: Session = Depends(utils.get_auth_db),
    auth=Depends(auth_handler),
):
    """Activate and deactivate an API Key."""
    email = auth[0]
    resp = apikey_crud.get_by_id(db=db, item_id=apikey_id, user=email)
    if not resp:
        raise HTTPException(status_code=404, detail="APIKey not found")

    if item.active == resp.active:
        return resp

    updated = apikey_crud.update(db=db, item_id=apikey_id, item=item)
    return updated


@router.delete(
    "/auth/api-keys",
    status_code=204,
)
async def delete_apikey(
    request: Request,
    apikey_id: uuid.UUID,
    db: Session = Depends(utils.get_auth_db),
    auth=Depends(auth_handler),
):
    """Delete an API Key."""
    email = auth[0]
    resp = apikey_crud.get_by_id(db=db, item_id=apikey_id, user=email)
    if not resp:
        raise HTTPException(status_code=404, detail="APIKey not found")

    apikey_crud.soft_delete(db=db, item_id=apikey_id, user=email)
    return {"message": "API-KEY deleted successfully."}
