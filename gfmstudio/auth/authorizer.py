# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import base64
import datetime
import functools
import json
from typing import Any, Optional, Union

import requests
from cachetools import TTLCache
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from redis.exceptions import ConnectionError
from sqlalchemy import func
from sqlalchemy.orm import Session

from gfmstudio.auth import utils
from gfmstudio.auth.api_key_utils import hash_api_key
from gfmstudio.auth.models import APIKey, User
from gfmstudio.auth.schemas import UserRequestSchema
from gfmstudio.common.api import crud
from gfmstudio.config import settings
from gfmstudio.log import logger
from gfmstudio.redis_client import redis_client

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
apikey_crud = crud.ItemCrud(model=APIKey)
user_crud = crud.ItemCrud(model=User)


AUTH_CACHE = TTLCache(maxsize=1024, ttl=60)


class CachedKeyData(BaseModel):
    email: str
    id: str
    active: bool
    expires_on: datetime.datetime


def get_redis_auth_key(api_key: str) -> str:
    """Returns the Redis key for a given plain API key."""
    return f"auth:api:{api_key}"


@functools.lru_cache
def get_auth_config(well_known_url: Optional[str] = None):
    well_known_url = (
        well_known_url
        or "https://geostudio.verify.ibm.com/oidc/endpoint/default/.well-known/openid-configuration"  # noqa: E501
    )
    response = requests.get(well_known_url)
    response.raise_for_status()
    return response.json()


def exchange_code_for_token(code, redirect_uri):
    token_endpoint = get_auth_config()["token_endpoint"]
    token_params = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": settings.OAUTH_CLIENTID,
        "client_secret": settings.OAUTH_CLIENTSECRET,
    }
    response = requests.post(token_endpoint, data=token_params)
    token_data = response.json()

    if error := token_data.get("error_description"):
        raise HTTPException(401, detail=error)

    access_token = token_data["access_token"]
    return access_token


def normalize_email(email: str | None) -> str | None:
    if email is None:
        return None
    return email.strip().lower()


async def get_api_key(
    api_key_value: Optional[str] = None,
    user_email: Optional[str] = None,
    db: Optional[Session] = None,
) -> Optional[list[dict[str, Any]]]:
    """Fetch API Key from data store.

    Parameters
    ----------
    api_key_value : str
        Value of the APIKey
    user_email : str
        User's email
    db : Session, optional
        Datasbase session, by default None

    Returns
    -------
    Optional[list[dict[str, Any]]]
        List of API key records or None if not found.

    """
    if not settings.AUTH_ENABLED:
        user_email = normalize_email(settings.DEFAULT_SYSTEM_USER)

    if (api_key_value or user_email) is None:
        raise TypeError(
            "missing required positional argument: `api_key_value` or `user_email`."
        )

    # 1. CACHE HIT (Fastest Path)
    api_key_hash: Optional[str] = None
    if api_key_value:
        api_key_hash = hash_api_key(api_key_value)
        cache_key = get_redis_auth_key(api_key_hash)
        try:
            if redis_client:
                cached_data_json = redis_client.get(cache_key)
                if cached_data_json:
                    return [json.loads(cached_data_json)]
        except (AttributeError, ConnectionError):
            logger.warning(
                "❌ Redis client not initialized ... Api-key cache lookup skipped."
            )

    # CACHE MISS (Expensive, triggers DB lookup and decryption)
    session = db or next(utils.get_auth_db())
    with session as db_session:
        if api_key_value and api_key_hash:
            resp = apikey_crud.get_all(
                db=db_session,
                filters={"hashed_key": api_key_hash},
                ignore_user_check=True,
            )

            # TODO: Remove this branch once the auth api-key database has been
            # populated with the hash of the api-key.
            if not resp:
                # Backward compatibility
                resp = apikey_crud.get_all(
                    db=db_session,
                    filters={"value": api_key_value},
                    ignore_user_check=True,
                )

            if resp:
                resp_obj = [
                    {
                        "email": item.user.email,
                        "id": item.id,
                        "value": item.value,
                        "active": item.active,
                        "expires_on": item.expires_on,
                    }
                    for item in resp
                ]
                cached_data = CachedKeyData(
                    email=resp_obj[0]["email"],
                    id=str(resp_obj[0]["id"]),
                    active=resp_obj[0]["active"],
                    expires_on=str(resp_obj[0]["expires_on"]),
                )
                cache_key = get_redis_auth_key(api_key_hash)

                try:
                    if redis_client:
                        redis_client.setex(cache_key, 60, cached_data.model_dump_json())
                except (AttributeError, ConnectionError):
                    logger.warning(
                        "❌ Redis client not initialized ... Api-key caching skipped."
                    )

                return resp_obj
        elif user_email:
            normalized_email = normalize_email(user_email)
            users = (
                db_session.query(User)
                .filter(
                    func.lower(User.email) == normalized_email,
                    User.deleted.isnot(True),
                )
                .order_by(User.created_at.desc())
                .all()
            )
            resp = users[0].apikeys if users else []
            email = users[0].email if users else normalized_email
            if resp:
                return [
                    {
                        "email": email,
                        "id": item.id,
                        "value": item.value,
                        "active": item.active,
                        "expires_on": item.expires_on,
                    }
                    for item in resp
                    if item.deleted is not True
                ]


def get_user_details(request: Request):
    try:
        user = request.headers["x-forwarded-email"]
    except KeyError:
        user = None
        logger.debug("Could not get user from oauth")

    try:
        token = request.headers["authorization"]
    except KeyError:
        token = None
        logger.debug("Could not get token from oauth")

    try:
        t = token.replace("Bearer ", "") if token else ""
        tC = json.loads(base64.b64decode(t.split(".")[1] + "=="))
        groups = tC.get("groupIds", [])
        user = tC["email"] if not user else user
    except:  # noqa: E722
        logger.debug("Could not get groups from oauth token")
        groups = []

    return user, token, groups


async def get_user_from_headers(request: Request):
    email, token, groups = get_user_details(request)

    if not settings.AUTH_ENABLED:
        email = settings.DEFAULT_SYSTEM_USER

    email = normalize_email(email)

    if email:
        names = email.split("@")[0]
        if len(names.split(".")) == 1:
            first_name, last_name = names.split(".")[0], names.split(".")[0]
        else:
            first_name, last_name = names.split(".")[0], names.split(".")[1]

        await load_user(
            user=UserRequestSchema(
                email=email,
                first_name=first_name,
                last_name=last_name,
            )
        )

    return email, token, groups


async def jwt_auth(
    request: Request,
    jwt_header: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
) -> tuple[Optional[str], Optional[str], list]:
    if not settings.AUTH_ENABLED:
        return normalize_email(settings.DEFAULT_SYSTEM_USER), None, []

    if jwt_header:
        email, token, groups = await get_user_from_headers(request=request)
        if email:
            return email, token, groups
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or Missing Authorization token.",
    )


async def api_key_auth(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
) -> tuple[Optional[str], Optional[str], list]:
    if not settings.AUTH_ENABLED:
        return normalize_email(settings.DEFAULT_SYSTEM_USER), None, []

    # Fetch API Key from data-store
    if api_key:
        api_key_obj = await get_api_key(api_key_value=api_key)
        if api_key_obj:
            return api_key_obj[0]["email"], api_key, []

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Expired or Missing API-Key.",
    )


async def load_user(user: UserRequestSchema, db: Optional[Session] = None):
    """Load new users to the datastore."""
    session = db or next(utils.get_auth_db())
    with session as db_session:
        normalized_email = normalize_email(user.email)
        if not normalized_email:
            return

        exiting_user = (
            db_session.query(User)
            .filter(
                func.lower(User.email) == normalized_email,
                User.deleted.isnot(True),
            )
            .order_by(User.created_at.desc())
            .all()
        )
        if not exiting_user:
            normalized_user = user.model_copy(update={"email": normalized_email})
            user_crud.create(
                db=db_session, item=normalized_user, user=normalized_email
            )


async def auth_handler(
    request: Request,
    jwt_header: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
    api_key: str = Security(api_key_header),
):
    try:
        resp = await jwt_auth(request=request)
        logger.debug("Authenticated (AuthToken): %s", resp[0])
        return resp
    except HTTPException:
        if request.headers.get("authorization"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization token is invalid or expired.",
            )

    try:
        resp = await api_key_auth(request=request, api_key=api_key)
        logger.debug("Authenticated (XApiKey): %s", resp[0])
        return resp
    except HTTPException:
        if request.headers.get("x-api-key"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="x-api-key provided is invalid or inactive.",
            )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Provide a valid Authorization or X-Api-Key header.",
    )
