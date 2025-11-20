# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from enum import Enum
from typing import Dict, Optional

from pydantic import AnyUrl, BaseModel, Field


class OnboardingStatus(str, Enum):
    PRESIGNED_URL_EXPIRED = "PRESIGNED_URL_EXPIRED"
    PRESIGNED_URL_FAILED = "PRESIGNED_URL_FAILED"
    ARTIFACT_TRANSFER_REQUEST_SUBMITTED = "ARTIFACT_TRANSFER_REQUEST_SUBMITTED"
    ARTIFACT_TRANSFER_STARTED = "ARTIFACT_TRANSFER_STARTED"
    ARTIFACT_TRANSFER_FAILED = "ARTIFACT_TRANSFER_FAILED"
    ARTIFACT_TRANSFER_COMPLETE = "ARTIFACT_TRANSFER_COMPLETE"
    MODEL_DEPLOY_REQUEST_SUBMITTED = "MODEL_DEPLOY_REQUEST_SUBMITTED"
    MODEL_DEPLOY_STARTED = "MODEL_DEPLOY_STARTED"
    MODEL_DEPLOY_FAILED = "MODEL_DEPLOY_FAILED"
    MODEL_DEPLOY_COMPLETE = "MODEL_DEPLOY_COMPLETE"
    MODEL_OFFBOARDING_REQUEST_SUBMITTED = "MODEL_OFFBOARDING_REQUEST_SUBMITTED"
    MODEL_OFFBOARDING_STARTED = "MODEL_OFFBOARDING_STARTED"
    MODEL_OFFBOARDING_FAILED = "MODEL_OFFBOARDING_FAILED"
    MODEL_OFFBOARDING_COMPLETE = "MODEL_OFFBOARDING_COMPLETE"

    def __str__(self) -> str:
        return self.value


class ModelFramework(str, Enum):
    TERRATORCH_V2 = "terratorch-v2"
    TERRATORCH_V1 = "terratorch"
    MMSEG = "mmseg"
    PYTORCH = "pytorch"

    def __str__(self) -> str:
        return self.value


class DeploymentType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    # AIU = "aiu", once AIU is determined

    def __str__(self) -> str:
        return self.value


class CPUResourceConfig(BaseModel):
    requests: Dict[str, str] = Field(
        default_factory=lambda: {"cpu": "6", "memory": "16G"},
        description="Requested resources",
        examples=[{"cpu": "6", "memory": "16G"}],
    )
    limits: Dict[str, str] = Field(
        default_factory=lambda: {"cpu": "12", "memory": "32G"},
        description="Resource limits",
        examples=[{"cpu": "12", "memory": "32G"}],
    )


class GPUResourceConfig(BaseModel):
    requests: Dict[str, str] = Field(
        default_factory=lambda: {"nvidia.com/gpu": "1"},
        description="Requested GPU resources",
        examples=[{"nvidia.com/gpu": "1"}],
    )
    limits: Dict[str, str] = Field(
        default_factory=lambda: {"nvidia.com/gpu": "1"},
        description="GPU resource limits",
        examples=[{"nvidia.com/gpu": "1"}],
    )


class OnboardModelRequest(BaseModel):
    model_framework: Optional[ModelFramework] = Field(
        default="terratorch",
        description="Options: [mmseg|terratorch|terratorch-v2|pytorch]",
    )
    model_id: str = Field(description="Identifier for the model")
    model_name: str = Field(description="Identifier for the model")
    model_configs_url: AnyUrl = Field(
        description="Presigned URL for the model configuration file"
    )
    model_checkpoint_url: AnyUrl = Field(
        description="Presigned URL for the model checkpoint file"
    )
    deployment_type: Optional[DeploymentType] = Field(
        default=DeploymentType.GPU, description="Deployment type: cpu, gpu"
    )

    resources: Optional[CPUResourceConfig] = Field(
        default_factory=CPUResourceConfig, description="CPU and memory resources"
    )
    gpu_resources: Optional[GPUResourceConfig] = Field(
        default_factory=GPUResourceConfig,
        description="GPU resources (only for GPU deployment)",
    )
    inference_container_image: Optional[str] = Field(
        default=None,
        description="Optional custom docker image to be used for deployment",
    )

    class Config:
        protected_namespaces = ()
