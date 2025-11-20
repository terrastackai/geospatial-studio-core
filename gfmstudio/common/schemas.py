# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, model_validator


class ItemResponse(BaseModel):
    id: uuid.UUID
    active: bool
    created_by: Optional[str] = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ListResponse(BaseModel):
    # Pagination fields
    total_records: Optional[int] = 0
    page_count: Optional[int] = None

    # Response data fields
    results: list[ItemResponse] = []

    class Config:
        from_attributes = True

    @model_validator(mode="before")
    def update_count(cls, values):
        """Update count

        Parameters
        ----------
        values : dict

        Returns
        -------
        dict
        """
        values["page_count"] = len(values["results"])
        return values
