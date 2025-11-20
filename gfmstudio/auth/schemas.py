# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import datetime
import uuid
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from gfmstudio.auth.api_key_utils import decrypt_key
from gfmstudio.common.schemas import ListResponse


class UserRequestSchema(BaseModel):
    first_name: Optional[str]
    last_name: Optional[str]
    email: str
    data_usage_consent: bool = Field(default=False)
    organization_id: Optional[str] = Field(default=None)
    extra_data: Optional[str] = Field(default=None)


class APIKeyRequestSchema(BaseModel):
    value: str
    hashed_key: str
    last_used_at: Optional[datetime.datetime] = Field(default=None)
    user_id: uuid.UUID


class APIKeyResponseSchema(BaseModel):
    id: uuid.UUID
    value: str
    expires_on: datetime.datetime
    active: bool

    @field_validator("value")
    def value_plain_text(cls, v):
        return decrypt_key(v)


class APIKeyListResponse(ListResponse):
    results: list[APIKeyResponseSchema]


class APIKeyUpdateSchema(BaseModel):
    active: bool
