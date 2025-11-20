# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

from gfmstudio.common.schemas import ItemResponse, ListResponse


class GeoDatasetTrainParamUpdateSchema(BaseModel):
    training_params: Optional[dict] = None

    @model_validator(mode="before")
    def custom_schema_validation(cls, values):
        training_params = values.get("training_params")
        if training_params:
            # Class weights should be defined for all classes or none.
            weights = training_params.get("class_weights", [])
            classes = training_params.get("classes", [])
            if weights and (len(weights) != len(classes)):
                raise ValueError(
                    "Class weights must either be defined for all classes or None"
                )
            if weights and not classes:
                raise ValueError("classes must be provided when defining class_weights")
        return values


class GeoDatasetMetadataUpdateSchema(BaseModel):
    dataset_name: Optional[str] = None
    description: Optional[str] = None
    custom_bands: Optional[List[dict]] = None
    label_categories: Optional[List[dict]] = None

    @field_validator("custom_bands")
    @classmethod
    def validate_custom_bands(cls, custom_bands):
        if custom_bands:
            for band in custom_bands:
                if band.get("id") == "":
                    raise ValueError("Valid band ID is needed")
        return custom_bands

    @field_validator("label_categories")
    @classmethod
    def validate_label_categories(cls, label_categories):
        if label_categories:
            for label_category in label_categories:
                if label_category.get("id") == "":
                    raise ValueError("Valid label category ID is needed")
        return label_categories


class GeoDatasetRequestSchema(GeoDatasetTrainParamUpdateSchema):
    dataset_name: str
    label_suffix: str
    dataset_url: str
    description: Optional[str]
    training_data_suffix: str
    purpose: Literal[
        "Regression", "Segmentation", "Generate", "NER", "Classify", "Other"
    ]
    custom_bands: List[dict] = []
    label_categories: Optional[List[dict]] = []


class GeoDatasetResponseSchema(ItemResponse):
    updated_by: Optional[str] = ""
    updated_at: Optional[datetime] = None
    dataset_name: str
    description: Optional[str] = ""
    dataset_url: Optional[str] = ""
    processed_data_url: Optional[str] = ""
    label_suffix: Optional[str] = ""
    training_data_suffix: Optional[str] = ""
    purpose: Optional[str] = ""
    custom_bands: Optional[Any]
    label_categories: Optional[Any]
    size: Optional[str] = ""
    status: Optional[str] = ""
    error: Optional[str] = ""


class GeoDatasetRequestSchemaV2(GeoDatasetTrainParamUpdateSchema):
    dataset_name: str
    label_suffix: str
    dataset_url: str
    description: Optional[str]
    purpose: Literal[
        "Regression", "Segmentation", "Generate", "NER", "Classify", "Other"
    ]
    data_sources: List[dict] = []
    label_categories: Optional[List[dict]] = []
    version: str = "v2"
    onboarding_options: Optional[dict] = {}


class GeoDatasetResponseSchemaV2(ItemResponse):
    id: str
    dataset_name: str
    description: Optional[str] = ""
    dataset_url: Optional[str] = ""
    label_suffix: Optional[str] = ""
    purpose: Optional[str] = ""
    data_sources: Optional[Any]
    label_categories: Optional[Any]
    size: Optional[str] = ""
    status: Optional[str] = ""
    error: Optional[str] = ""
    logs: Optional[str] = ""
    onboarding_options: Optional[dict] = {}


class DatasetSummaryResponseSchema(BaseModel):
    id: str
    dataset_name: str
    description: Optional[str] = ""

    class Config:
        from_attributes = True
        extra = "ignore"


class GeoDatasetPreScanRequestSchema(BaseModel):
    dataset_url: str
    label_suffix: str
    training_data_suffix: str


class GeoDatasetPreScanRequestSchemaV2(BaseModel):
    dataset_url: str
    label_suffix: str
    training_data_suffixes: List[str]


class GeoDatasetsResponseSchema(ListResponse):
    results: List[GeoDatasetResponseSchema]


class GeoDatasetsResponseSchemaV2(ListResponse):
    results: List[GeoDatasetResponseSchemaV2]


class DatasetsSummaryResponseSchema(ListResponse):
    results: List[DatasetSummaryResponseSchema]
