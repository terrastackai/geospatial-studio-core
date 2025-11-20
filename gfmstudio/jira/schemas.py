# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from gfmstudio.common.schemas import ListResponse


class IssueTypeResponseSchema(BaseModel):
    name: str


class StatusResponseSchema(BaseModel):
    name: str


class AuthorResponseSchema(BaseModel):
    displayName: str

    @field_validator("displayName")
    def check_displayName(cls, v):
        if v in ["watsonx.geo watsonx.geo"]:
            return "You"
        return v


class CommentResponseSchema(BaseModel):
    id: str
    body: str
    author: AuthorResponseSchema
    created: datetime.datetime
    updated: datetime.datetime


class CommentListResponse(BaseModel):
    comments: list[CommentResponseSchema]


class FieldsResponseSchema(BaseModel):
    issuetype: IssueTypeResponseSchema
    summary: str
    created: datetime.datetime
    updated: datetime.datetime
    duedate: Optional[datetime.datetime]
    status: StatusResponseSchema
    description: str
    comment: CommentListResponse


class IssueResponseSchema(BaseModel):
    id: str
    key: str
    fields: FieldsResponseSchema


class IssueListResponse(ListResponse):
    results: list[IssueResponseSchema]


class IssueRequestSchema(BaseModel):
    issuetype: str = Field(
        description="The type of report, e.g. Bug, New Feature, Feadback",
        example="Task",
    )
    summary: str = Field(
        description="A summary of the issue",
        example="Summary of the issue...",
    )
    description: str = Field(
        description="Detailed account of the issue",
        example="Detailed description of the issue...",
    )


class CommentRequestSchema(BaseModel):
    body: str = Field(
        description="Comment text",
        example="This is my comment...",
    )
