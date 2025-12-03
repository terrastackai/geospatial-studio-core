# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from enum import Enum
from typing import Any, Literal, Optional, Union

import structlog
from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from gfmstudio.config import settings

logger = structlog.get_logger()


# ===============
# Enum Options
# ===============
class LRPolicyName(str, Enum):
    """Learning Rate Policy Enums

    Attributes
    ----------
    Fixed : str
        Fixed Learning Rate Policy
    Cosine : str
        Cosine Learning Rate Policy

    Methods
    -------
    __str__
        Returns a string representation of learning Rate Policy.
    """

    Fixed = "Fixed"
    Cosine = "Cosine"

    def __str__(self) -> str:
        """Generates string representation of the Learning Rate Policy

        Returns
        -------
        str
            String representation of Learning Rate Policy value
        """
        return self.value


class NecksType(str, Enum):
    """Necks Enums

    Attributes
    ----------
    PermuteDims : str
        PermuteDims Necks
    InterpolateToPyramidal : str
        InterpolateToPyramidal Necks
    MaxpoolToPyramidal : str
        MaxpoolToPyramidal Necks
    AddBottleneckLayer : str
        AddBottleneckLayer Necks
    ReshapeTokensToImage : str
        ReshapeTokensToImage Necks
    LearnedInterpolateToPyramidal : str
        LearnedInterpolateToPyramidal Necks
    Methods
    -------
    __str__
        Returns a string representation of Necks.
    """

    PermuteDims = "PermuteDims"
    InterpolateToPyramidal = "InterpolateToPyramidal"
    MaxpoolToPyramidal = "MaxpoolToPyramidal"
    AddBottleneckLayer = "AddBottleneckLayer"
    ReshapeTokensToImage = "ReshapeTokensToImage"
    LearnedInterpolateToPyramidal = "LearnedInterpolateToPyramidal"

    def __str__(self) -> str:
        """Generates string representation of the Necks

        Returns
        -------
        str
            String representation of Necks value
        """
        return self.value


class LossFunctionType(str, Enum):
    """Loss Function Types Enum

    Attributes
    ----------
    MSELoss : str
        Mean Squared Error Loss
    RMSELoss : str
        Root Mean Squared Error Loss
    CrossEntropyLoss : str
        CrossEntropy Loss
    HuberLoss : str
        Huber Loss

    #TODO: Options to support any changes in templates.
    L1Loss : str
        L1 Loss
    CTCLoss : str
        CTC Loss
    NLLLoss : str
        NLL Loss
    PoissonNLLLoss : str
        Poisso NLL Loss
    GaussianNLLLoss : str
        Gaussian NLL Loss
    KLDivLoss : str
        KLDiv Loss
    BCELoss : str
        BCELoss
    BCEWithLogitsLoss : str
        BCE With Logits Loss
    MarginRankingLoss : str
        Margin Ranking Loss
    HingeEmbeddingLoss : str
        Hinge Embedding Loss
    MultiLabelMarginLoss : str
        Multi Label Margin Loss
    SmoothL1Loss : str
        Smooth L1 Loss
    SoftMarginLoss : str
        Soft Margin Loss
    MultiLabelSoftMarginLoss : str
        Multi Label SoftMargin Loss
    CosineEmbeddingLoss : str
        Cosine Embedding Loss
    MultiMarginLoss : str
        MultiMargin Loss
    TripletMarginLoss : str
        Triplet Margin Loss
    TripletMarginWithDistanceLoss : str
        Triplet Margin With Distance Loss

    Methods
    -------
    __str__
        Returns a string representation of Loss Function Types.
    """

    MSELoss = "mse"
    RMSELoss = "rmse"
    CrossEntropyLoss = "ce"
    HuberLoss = "huber"
    MeanAbsoluteError = "mae"

    # Options to support any changes in templates.
    # L1Loss = "L1Loss"
    # CTCLoss = "CTCLoss"
    # NLLLoss = "NLLLoss"
    # PoissonNLLLoss = "PoissonNLLLoss"
    # GaussianNLLLoss = "GaussianNLLLoss"
    # KLDivLoss = "KLDivLoss"
    # BCELoss = "BCELoss"
    # BCEWithLogitsLoss = "BCEWithLogitsLoss"
    # MarginRankingLoss = "MarginRankingLoss"
    # HingeEmbeddingLoss = "HingeEmbeddingLoss"
    # MultiLabelMarginLoss = "MultiLabelMarginLoss"
    # HuberLoss = "HuberLoss"
    # SmoothL1Loss = "SmoothL1Loss"
    # SoftMarginLoss = "SoftMarginLoss"
    # MultiLabelSoftMarginLoss = "MultiLabelSoftMarginLoss"
    # CosineEmbeddingLoss = "CosineEmbeddingLoss"
    # MultiMarginLoss = "MultiMarginLoss"
    # TripletMarginLoss = "TripletMarginLoss"
    # TripletMarginWithDistanceLoss = "TripletMarginWithDistanceLoss"

    def __str__(self) -> str:
        """Generates string representation of the Loss Function Type

        Returns
        -------
        str
            String representation of Loss Function Type value
        """
        return self.value


class OptimizerType(str, Enum):
    """Optimizer Type Enums

    Attributes
    ----------
    Adam : str
        Adam Optimizer Type
    SGD : str
        Cosine Optimizer Type
    AdamW : str
        AdamW Optimizer Type
    RMSProp : str
        RMSProp Optimizer Type

    #TODO: Options to support any changes in templates.
    Adadelta : str
        Adadelta Optimizer Type
    Adagrad : str
        Adagrad Optimizer Type
    SparseAdam : str
        SparseAdam Optimizer Type
    Adamax : str
        Adamax Optimizer Type
    ASGD : str
        ASGD Optimizer Type
    LBFGS : str
        LBFGS Optimizer Type
    NAdam : str
        NAdam Optimizer Type
    RAdam : str
        RAdam Optimizer Type
    Rprop : str
        Rprop Optimizer Type

    Methods
    -------
    __str__
        Returns a string representation of Optimizer Type.
    """

    Adam = "Adam"
    SGD = "SGD"
    AdamW = "AdamW"
    RMSProp = "RMSProp"

    # Options to support any changes in templates.
    # Adadelta = "Adadelta"
    # Adagrad = "Adagrad"
    # SparseAdam = "SparseAdam"
    # Adamax = "Adamax"
    # ASGD = "ASGD"
    # LBFGS = "LBFGS"
    # NAdam = "NAdam"
    # RAdam = "RAdam"
    # Rprop = "Rprop"

    def __str__(self) -> str:
        """Generates string representation of the Optimizer Type.

        Returns
        -------
        str
            String representation of Optimizer Type value.
        """
        return self.value


class LRPolicy(BaseModel):
    """A model for learning rate policies and configurations.

    Attributes
    ----------
    policy : Optional[str]
        The name of the learning rate policy. Defaults to "Fixed".
    warmup_iters : Optional[int]
        The number of warmup iterations. This is valid for some learning rate policies.
        Defaults to 1000.
    warmup_ratio : Optional[float]
        The initial learning rate at warmup will be set to learning_rate * warmup_ratio.
        Defaults to 1.0.

    Examples
    --------
    Create an LRPolicy object:

    >>> lr_policy_custom = LRPolicy(policy="StepLR", warmup_iters=500, warmup_ratio=0.1)
    >>> print(lr_policy_custom)
    LRPolicy(policy='StepLR', warmup_iters=500, warmup_ratio=0.1)
    """

    policy: Optional[str] = Field(default="Fixed", description="Policy name")
    warmup_iters: Optional[int] = Field(
        default=1000,
        description="LR warmup iterations. Valid for some policies",
    )
    warmup_ratio: Optional[float] = Field(
        description="Initial lr at warmup will be learning_rate * warmup_ratio",
        default=1,
    )

    @field_validator("warmup_iters", mode="before")
    def check_non_negative_warmup_iters(cls, value):
        """field_validator to ensure warmup_iters is non-negative."""
        if value is not None and value < 0:
            raise ValueError(
                status_code=422, detail="warmup_iters must be non-negative"
            )
        return value

    @field_validator("warmup_ratio", mode="before")
    def check_non_negative_warmup_ratio(cls, value):
        """Validator to ensure warmup_ratio is non-negative."""
        if value is not None and value < 0:
            raise ValueError(
                status_code=422, detail="warmup_ratio must be non-negative"
            )
        return value


class LossFunction(BaseModel):
    """
    A model for specifying loss function configuration in machine learning models.

    Attributes
    ----------
    type : Optional[str]
        The type of loss function to be used. Defaults to "CrossEntropyLoss".
    avg_non_ignore : bool
        If True, the loss is averaged only over non-ignored targets,
        where the labels are present. Ignored targets (e.g., missing labels)
        are excluded from the averaging. Defaults to True.

    Examples
    --------
    Create a LossFunction object:

    >>> custom_loss_fn = LossFunction(type="MSELoss", avg_non_ignore=False)
    >>> print(custom_loss_fn)
    LossFunction(type='MSELoss', avg_non_ignore=False)
    """

    type: Optional[str] = Field(
        default="CrossEntropyLoss", description="Type of loss function"
    )
    avg_non_ignore: bool = Field(
        default=True,
        description=(
            "The loss is only averaged over non-ignored targets "
            "(ignored targets are usually where labels are missing in the dataset) if this is True"
        ),
    )


class AuxLossFunction(LossFunction):
    """
    A model for auxiliary loss function configuration, inheriting from LossFunction.

    Attributes
    ----------
    loss_weight : Optional[float]
        The weight to apply to the auxiliary loss.
        This controls the contribution of this loss to the total loss. Defaults to 0.2.

    Examples
    --------
    Create an AuxLossFunction object:

    >>> custom_aux_loss_fn = AuxLossFunction(type="MSELoss", avg_non_ignore=False, loss_weight=0.5)
    >>> print(custom_aux_loss_fn)
    AuxLossFunction(type='MSELoss', avg_non_ignore=False, loss_weight=0.5)
    """

    loss_weight: Optional[float] = Field(default=1.0)


# ===============
# Dataset Params
# ===============
class ModelDatasetParams(BaseModel):
    """
    Parameters for configuring the model dataset. Provided when onboarding data

    Attributes
    ----------
    num_workers : Optional[int]
        Number of workers for data loading. Defaults to 2.
    num_frames : Optional[int]
        Number of frames per dataset sample. Defaults to 1.
    img_suffix : Optional[str]
        Suffix for image files. Defaults to None.
    seg_map_suffix : Optional[str]
        Suffix for segmentation map files. Defaults to None.
    regression : Optional[bool]
        If True, indicates the dataset is used for regression tasks. Defaults to False.
    bands : Optional[List[Any]]
        List of bands available in the dataset. Defaults to None.
    output_bands : Optional[List[Any]]
        List of output bands to fine-tune. Defaults to None.
    rgb_band_indices : Optional[List[int]]
        Indices of RGB bands, assuming a 3-band image. Defaults to [2, 1, 0].
    train_split_path : Optional[str]
        Path to the train split file. Defaults to "other/flood_train_data_S2.txt".
    val_split_path : Optional[str]
        Path to the validation split file. Defaults to "other/flood_valid_data_S2.txt".
    test_split_path : Optional[str]
        Path to the test split file. Defaults to "other/flood_test_data_S2.txt".
    train_data_dir : Optional[str]
        Directory containing training data. Defaults to an empty string.
    train_labels_dir : Optional[str]
        Directory containing training labels. Defaults to an empty string.
    test_data_dir : Optional[str]
        Directory containing test data. Defaults to an empty string.
    test_labels_dir : Optional[str]
        Directory containing test labels. Defaults to an empty string.
    val_data_dir : Optional[str]
        Directory containing validation data. Defaults to "training-data".
    val_labels_dir : Optional[str]
        Directory containing validation labels. Defaults to "labels".
    ignore_index : Optional[str]
        Index to ignore during evaluation. Defaults to "-1".
    constant_multiply : Optional[float]
        Constant to multiply data by, for normalization or scaling purposes. Defaults to "0.0001".
    classes : Optional[List]
        List of class labels. Defaults to [0, 1].
    class_weights : Optional[List]
        List of class weights for loss calculation. Defaults to an empty list.
    norm_means : Optional[List[float]]
        List of means for normalization. Defaults to an empty list.
    norm_stds : Optional[List[float]]
        List of standard deviations for normalization. Defaults to an empty list.
    orig_img_size : Optional[int]
        Original image size. Defaults to 512.
    label_nodata : Optional[str]
        Value representing 'no data' in the label. Defaults to "-1".
    image_nodata : Optional[str]
        Value representing 'no data' in the image. Defaults to "-9999".
    image_nodata_replace : Optional[str]
        Value to replace 'no data' in the image. Defaults to "0".
    image_to_float : Optional[bool]
        If True, converts the image data to float type. Defaults to False.
    orig_image_size : Optional[int]
        Original size of the input images. Defaults to 512.
    batch_size : Optional[int]
        Batch size for data loading. Defaults to 4.

    Examples
    --------
    Create a ModelDatasetParams object:

    >>> params = ModelDatasetParams()
    >>> print(params)
    ModelDatasetParams(num_workers=2, num_frames=1, img_suffix=None, seg_map_suffix=None,
    regression=False, bands=None, output_bands=None, rgb_band_indices=[2, 1, 0],
    train_split_path='other/flood_train_data_S2.txt', val_split_path='other/flood_valid_data_S2.txt',
    test_split_path='other/flood_test_data_S2.txt', train_data_dir='', train_labels_dir='',
    test_data_dir='', test_labels_dir='', val_data_dir='training-data', val_labels_dir='labels',
    ignore_index='-1', constant_multiply=0.0001, classes=[0, 1], class_weights=[],
    norm_means=[], norm_stds=[], orig_img_size=512, label_nodata=-1, image_nodata='-9999',
    image_nodata_replace='0', image_to_float=False, orig_image_size=512, batch_size=4)
    """

    # not user defined
    num_workers: Optional[int] = Field(description="", default=2)

    # from dataset
    num_frames: Optional[int] = Field(description="", default=1)
    img_suffix: Optional[dict[str, str]] = Field(
        description="dict of lists of image suffixes for each image modality",
        default=None,
    )
    seg_map_suffix: Optional[str] = Field(description="", default=None)
    regression: Optional[bool] = Field(description="", default=False)
    bands: Optional[dict[str, list]] = Field(
        description="dict of lists of dataset bands for each image modality",
        default=None,
    )
    output_bands: Optional[dict[str, list[Any]]] = Field(
        description="dict of lists of output bands for each image modality",
        default=None,
    )
    rgb_band_indices: Optional[list[int]] = [2, 1, 0]
    train_split_path: Optional[str] = Field(
        description="", default="other/flood_train_data_S2.txt"
    )
    val_split_path: Optional[str] = Field(
        description="", default="other/flood_valid_data_S2.txt"
    )
    test_split_path: Optional[str] = Field(
        description="", default="other/flood_test_data_S2.txt"
    )
    train_data_dir: Optional[dict[str, str]] = Field(description="", default="")
    train_labels_dir: Optional[str] = Field(description="", default="")
    test_data_dir: Optional[dict[str, str]] = Field(description="", default="")
    test_labels_dir: Optional[str] = Field(description="", default="")
    val_data_dir: Optional[dict[str, str]] = Field(
        description="", default="training-data"
    )
    val_labels_dir: Optional[str] = Field(description="", default="labels")
    ignore_index: Optional[str] = Field(description="", default="-1")
    constant_multiply: Optional[float] = Field(description="", default=1.0)
    classes: Optional[list] = [0, 1]
    class_weights: Optional[list] = []
    norm_means: Optional[dict[str, list]] = Field(
        description="dict of lists of normalised means for each image modality",
        default=None,
    )
    norm_stds: Optional[dict[str, list]] = Field(
        description="dict of lists of means for each image modality", default=None
    )
    orig_img_size: Optional[int] = Field(description="", default=512)
    label_nodata: Optional[int] = Field(description="", default=-1)
    image_nodata: Optional[int] = Field(description="", default=-9999)
    image_nodata_replace: Optional[int] = Field(
        description="", default=0
    )  # maybe user can override?
    image_to_float: Optional[bool] = Field(description="", default=False)
    orig_image_size: Optional[int] = Field(description="", default=512)
    batch_size: Optional[int] = Field(description="", default=2)
    num_modalities: Optional[int] = Field(
        description="Total number of image modalities in the dataset.", default=1
    )
    image_modalities: Optional[list[str]] = Field(
        description="The image modalities in the dataset.", default=[]
    )
    rgb_modality: Optional[str] = Field(
        description="Image Modality that has the RGB bands.", default=None
    )


# ====================
# User Defined Params
# ====================
class DataTrainTrasnsform(BaseModel):
    class_path: str
    height: Optional[int] = None
    width: Optional[int] = None
    always_apply: Optional[bool] = False
    transpose_mask: Optional[bool] = None
    p: Optional[int] = 1.0


class DataLoading(BaseModel):
    """
    Parameters for configuring the data loading process when fine-tuning.

    Attributes
    ----------
    batch_size : Optional[int]
        The number of samples in each batch. Defaults to None.
    bands : Optional[List[Any]]
        List of bands to load for each data sample. Defaults to None.
    workers_per_gpu : Optional[int]
        Number of data loader workers per GPU. Defaults to None.
    random_flip : Optional[int]
        Probability of flipping images during data augmentation (0 for no flip, 1 for always flip). Defaults to 0.
    tuning_bands: Optional[List]
        Which bands do you want to fine-tune the backbone with. These bands are subset of bands.
    constant_multiply: Optional[float]
        Constant to multiply data by, for normalization or scaling purposes. Defaults to None.


    Examples
    --------
    Create a DataLoading object:

    >>> custom_data_loading = DataLoading(
            batch_size=16, workers_per_gpu=4, random_flip=1,
            bands=['RED', 'GREEN', 'BLUE'], tuning_bands=['GREEN', 'BLUE'])
    >>> print(custom_data_loading)
    DataLoading(
        batch_size=16, bands=['RED', 'GREEN', 'BLUE'],
        workers_per_gpu=4, random_flip=1, tuning_bands=['GREEN', 'BLUE'])
    """

    batch_size: Optional[int] = None
    bands: Optional[list[Any]] = Field(
        description="Bands available in the dataset", default=None
    )
    workers_per_gpu: Optional[int] = None
    random_flip: Optional[int] = 0
    tuning_bands: Optional[list] = Field(
        description="Bands to fine-tune.", default=None
    )
    train_transform: Optional[list[DataTrainTrasnsform]] = None
    drop_last: Optional[bool] = None
    expand_temporal_dimension: Optional[bool] = None
    ignore_index: Optional[str] = None
    constant_multiply: Optional[float] = None
    classes: Optional[list] = None
    class_weights: Optional[list] = None
    norm_means: Optional[list[float]] = None
    norm_stds: Optional[list[float]] = None
    model_config = {"extra": "allow"}
    


class Runner(BaseModel):
    """
    A class to configure the training runner parameters.

    Attributes
    ----------
    max_epochs : Optional[int]
        The maximum number of training epochs. Defaults to 10.
    early_stopping_patience : Optional[int]
        The number of epochs to wait for improvement before stopping training early.
        If None, early stopping is disabled.
    early_stopping_monitor : Optional[Literal["val/loss"]]
        The metric to monitor for early stopping. Currently only supports "val/loss".
    plot_on_val: Optional[Union[bool, int]]
        Whether to plot visualizations on validation.
            If true, log every epoch. Defaults to 10. If int, will plot every plot_on_val epochs.

    Raises
    ------
    ValueError
        If early_stopping_monitor is defined but early_stopping_patience is None.

    Examples
    --------
    >>> runner = Runner(max_epochs=20, early_stopping_patience=5, early_stopping_monitor="val/loss", plot_on_val=False)
    >>> print(runner)
    Runner(max_epochs=20, early_stopping_patience=5, early_stopping_monitor='val/loss', plot_on_val=False)

    >>> runner_no_early_stopping = Runner(max_epochs=15)
    >>> print(runner_no_early_stopping)
    Runner(max_epochs=15, early_stopping_patience=None, early_stopping_monitor=None,plot_on_val=False)

    >>> # This will raise a ValueError
    >>> invalid_runner = Runner(max_epochs=10, early_stopping_monitor="val/loss")
    """

    max_epochs: Optional[int] = 10
    early_stopping_patience: Optional[int] = None
    early_stopping_monitor: Optional[Literal["val/loss"]] = None
    plot_on_val: Optional[Union[bool, int]] = 2

    @model_validator(mode="before")
    @classmethod
    def check_early_stopping(cls, values):
        if values.get("early_stopping_monitor") and values.get(
            "early_stopping_patience"
        ):
            return values
        elif not values.get("early_stopping_monitor") and not values.get(
            "early_stopping_patience"
        ):
            return values
        raise ValueError(
            status_code=422,
            detail="Set both early_stopping_patience and early_stopping_monitor or None of them.",
        )


class Optimizer(BaseModel):
    """
    A class to configure optimizer parameters for training.

    Attributes
    ----------
    type : Optional[str]
        The type of optimizer to use. Defaults to "Adam".
    lr : Optional[str]
        The learning rate for the optimizer. Should be provided as a string to support
        dynamic configurations.
    weight_decay : Optional[float]
        The weight decay (L2 regularization) coefficient. If None, no weight decay is applied.

    Examples
    --------
    >>> optimizer = Optimizer(type="SGD", lr="0.01", weight_decay=0.001)
    >>> print(optimizer)
    Optimizer(type='SGD', lr='0.01', weight_decay=0.001)

    >>> default_optimizer = Optimizer()
    >>> print(default_optimizer)
    Optimizer(type='Adam', lr=None, weight_decay=None)

    """

    type: Optional[str] = Field(description="Optimizer", default="AdamW")
    lr: Optional[float] = None
    weight_decay: Optional[float] = None

    class Config:
        coerce_numbers_to_str = True


class TiledInferenceParameters(BaseModel):
    """A class to configure Tiled inference parameters for inference.
            The stride is a bit less than h_crop as it allows for taking care
            of boundary between tiles better.

    Attributes
    ----------
    h_crop : int
        The height to crop / tile the inference image
    h_stride : int
        The height stride to slide between image patches.

    w_crop : int
        The width to crop / tile the inference image
    w_stride : int
        The width stride to slide between image patches
    average_patches: bool
        Whether to average the image patches
    """

    h_crop: Optional[int] = Field(default=None, gt=0)
    h_stride: Optional[int] = Field(default=None, gt=0)
    w_crop: Optional[int] = Field(default=None, gt=0)
    w_stride: Optional[int] = Field(default=None, gt=0)
    average_patches: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def check_all_or_none(cls, model):

        relevant_fields = [
            "h_crop",
            "h_stride",
            "w_crop",
            "w_stride",
            "average_patches",
        ]
        values_list = [model.get(field) for field in relevant_fields]

        all_none = all(v is None for v in values_list)
        all_set = all(v is not None for v in values_list)

        if all_none or all_set:
            return model
        else:
            raise ValueError(
                status_code=422,
                detail="Set all h_crop, h_stride, w_crop, w_stride and average_patches values or None of them.",
            )


class Evaluation(BaseModel):
    """
    Evaluation settings for model training.

    Attributes
    ----------
    interval : Optional[int]
        Frequency (in epochs) with which validation is performed. Defaults to 1.
    metric : Optional[str]
        Metric used for evaluation. Defaults to 'mIoU' (mean Intersection over Union).

    Examples
    --------
    Create an Evaluation object:

    >>> custom_eval = Evaluation(interval=5, metric='accuracy')
    >>> print(custom_eval)
    Evaluation(interval=5, metric='accuracy')
    """

    interval: Optional[int] = Field(
        default=1,
        ge=0,
        description="Frequency of epochs with which to perform validation",
    )
    metric: Optional[str] = Field(default="mIoU")


class DecodeHead(BaseModel):
    """
    The decode head for a neural network model, typically used for image segmentation tasks.

    Attributes
    ----------
    channels : Optional[int]
        Number of channels at each block of the auxiliary head, except the final one. Defaults to 32.
    num_convs : Optional[int]
        Number of convolutional blocks in the decode head, excluding the final block. Defaults to 1.
    loss_decode : Optional[LossFunction]
        Defines the loss function to be applied during decoding. Defaults to a LossFunction object.

    Examples
    --------
    Create a DecodeHead object:

    >>> custom_decode_head = DecodeHead(channels=64, num_convs=3, loss_decode=LossFunction(type='DiceLoss'))
    >>> print(custom_decode_head)
    DecodeHead(channels=64, num_convs=3, loss_decode=LossFunction(type='DiceLoss', avg_non_ignore=True))
    """

    decoder: Optional[str] = "UperNetDecoder"
    channels: Optional[int] = Field(
        default=256,
        description="Channels at each block of the aux head, except the final one",
    )
    num_convs: Optional[int] = Field(
        default=None,
        description="Number of convolutional blocks in the head (except the final one)",
    )
    loss_decode: Optional[LossFunction] = LossFunction()
    backbone_num_frames: Optional[int] = Field(
        default=None, description="Setup bitemporal sampling for the model"
    )


class BaseModelNecks(BaseModel):
    name: NecksType = Field(
        default=NecksType.ReshapeTokensToImage,
        description="neck to transform output to what the decoder can ",
    )


class AuxiliaryHead(DecodeHead):
    """
    Auxiliary head used in a neural network model, extending the main decode head,
    and including an auxiliary loss function.

    Attributes
    ----------
    channels : Optional[int]
        Number of channels at each block of the auxiliary head, except the final one. Inherited from `DecodeHead`.
        Defaults to 32.
    num_convs : Optional[int]
        Number of convolutional blocks in the auxiliary head, excluding the final block. Inherited from `DecodeHead`.
        Defaults to 1.
    loss_decode : Optional[AuxLossFunction]
        Defines the auxiliary loss function used for training.
        Defaults to an `AuxLossFunction` object with a loss weight of 0.2.

    Examples
    --------
    Create a `AuxiliaryHead` object:

    >>> custom_aux_head = AuxiliaryHead(
            channels=64, num_convs=2, loss_decode=AuxLossFunction(type='DiceLoss', loss_weight=0.3))
    >>> print(custom_aux_head)
    AuxiliaryHead(
        channels=64,
        num_convs=2,
        loss_decode=AuxLossFunction(type='DiceLoss', avg_non_ignore=True, loss_weight=0.3)
    )
    """

    decoder: Optional[str] = "FCNDecoder"
    loss_decode: Optional[AuxLossFunction] = AuxLossFunction()
    in_index: Optional[int] = -1
    dropout: Optional[float] = 0


class Model(BaseModel):
    """
    A class to configure the architecture of a machine learning model.

    Attributes
    ----------
    frozen_backbone : Optional[bool]
        A flag indicating whether the backbone of the model should be frozen during training.
        Defaults to False, allowing training of all layers.
    decode_head : Optional[DecodeHead]
        An instance of the DecodeHead class representing the decoding head of the model.
        If not provided, defaults to a new instance of DecodeHead.
    auxiliary_head : Optional[AuxiliaryHead]
        An instance of the AuxiliaryHead class representing an auxiliary head for additional
        tasks, if applicable. Defaults to None if not used.
    tiled_inference_parameters:Optional[TiledInferenceParameters]

    Examples
    --------
    >>> model = Model(frozen_backbone=True)
    >>> print(model)
    Model(frozen_backbone=True, decode_head=DecodeHead(...), auxiliary_head=None)

    >>> decode_head_instance = DecodeHead(...)  # Replace with actual instantiation
    >>> model_with_heads = Model(decode_head=decode_head_instance, auxiliary_head=AuxiliaryHead())
    >>> print(model_with_heads)
    Model(frozen_backbone=False, decode_head=DecodeHead(...), auxiliary_head=AuxiliaryHead(...))
    """

    frozen_backbone: Optional[bool] = False
    decode_head: Optional[DecodeHead] = DecodeHead()
    auxiliary_head: Optional[AuxiliaryHead] = None
    optimizer: Optional[Optimizer] = Field(
        description="Model specific optimizer", default=None
    )
    tiled_inference_parameters: Optional[TiledInferenceParameters] = None
    necks: Optional[list[BaseModelNecks]] = []
    backbone_img_size: Optional[int] = None


class TemplateUserDefinedParams(BaseModel):
    """
    A class for defining user-configurable parameters for a model training template.

    Attributes
    ----------
    dataset_id : Optional[str]
        The dataset id. Defaults to None.
    data : Optional[DataLoading]
        Parameters for configuring data loading, including batch size and number of workers.
        Defaults to a DataLoading object.
    runner : Optional[Runner]
        Defines training parameters such as the number of epochs and early stopping criteria.
        Defaults to a Runner object.
    optimizer : Optional[Optimizer]
        The optimizer configuration, including type (e.g., Adam) and learning rate settings.
        Defaults to an Optimizer object.
    lr_config : Optional[LRPolicy]
        Learning rate policy configuration, including warmup settings. Defaults to None.
    evaluation : Optional[Evaluation]
        Parameters for model evaluation, such as validation frequency and metric. Defaults to an Evaluation object.
    model : Optional[Model]
        The model configuration, including whether to freeze the backbone and the
        definition of the decode and auxiliary heads. Defaults to a Model object.
    backbone_model_id : Optional[str]
        Base model id. Defaults to None.



    Examples
    --------
    Create a `TemplateUserDefinedParams` object:

    >>> custom_params = TemplateUserDefinedParams(
    ...     dataset_id="dataset123",
    ...     runner=Runner(max_epochs=50),
    ...     optimizer=Optimizer(type='SGD', lr="0.01"),
    ...     evaluation=Evaluation(interval=5, metric='accuracy'),
    ...     model=Model(frozen_backbone=True)
    ... )
    >>> print(custom_params)
    TemplateUserDefinedParams(
        dataset_id='dataset123',
        data=DataLoading(batch_size=None, bands=None, workers_per_gpu=None, random_flip=0),
        runner=Runner(max_epochs=50, early_stopping_patience=None, early_stopping_monitor=None, plot_on_val=False),
        optimizer=Optimizer(type='SGD', lr='0.01', weight_decay=None),
        lr_config=None,
        evaluation=Evaluation(interval=5, metric='accuracy'),
        model=Model(frozen_backbone=True, decode_head=DecodeHead(
            channels=32, num_convs=1, loss_decode=LossFunction(
                type='CrossEntropyLoss', avg_non_ignore=True)), auxiliary_head=None),
        backbone_model_id=None
    )
    """

    dataset_id: Optional[str] = None
    data: Optional[DataLoading] = DataLoading()
    runner: Optional[Runner] = Runner()
    optimizer: Optional[Optimizer] = Optimizer()
    lr_config: Optional[LRPolicy] = None
    evaluation: Optional[Evaluation] = Evaluation()
    model: Optional[Model] = Model()
    backbone_model_id: Optional[str] = None
    model_config = {"extra": "allow"}


class ModelBaseParams(BaseModel):
    """
    A class to define the base parameters for a model architecture.

    Attributes
    ----------
    num_layers : Optional[int]
        The number of layers in the model. Defaults to 12.
    patch_size : Optional[int]
        The size of the patches to be processed by the model. Defaults to 16.
    embed_dim : Optional[int]
        The dimensionality of the embedding space. Defaults to 768.
    num_heads : Optional[int]
        The number of attention heads in the model. Defaults to 12.
    tubelet_size : Optional[int]
        The size of the tubelets (if applicable). Defaults to 1.
    tile_size : Optional[int]
        The size of the tiles used in processing. Defaults to 224.
    head_channel_list : Optional[List[int]]
        A list of channels for each head in the model. Defaults to [256].
    pretrained_weights_path : Optional[str]
        The file path to the pretrained weights. Defaults to "/files/pre-trained/...".

    Examples
    --------
    >>> params = ModelBaseParams()
    >>> print(params)
    ModelBaseParams(num_layers=12, patch_size=16, embed_dim=768, num_heads=12,
                     tubelet_size=1, tile_size=224, head_channel_list=[256],
                     pretrained_weights_path='/files/pre-trained/...')

    >>> custom_params = ModelBaseParams(num_layers=24, embed_dim=512,
    ...                                  head_channel_list=[128, 256])
    >>> print(custom_params)
    ModelBaseParams(num_layers=24, patch_size=16, embed_dim=512, num_heads=12,
                     tubelet_size=1, tile_size=224, head_channel_list=[128, 256],
                     pretrained_weights_path='/files/pre-trained/...')

    """

    # base model dependent
    num_layers: Optional[int] = Field(description="", default=12)
    patch_size: Optional[int] = None
    embed_dim: Optional[int] = Field(description="", default=768)
    num_heads: Optional[int] = Field(description="", default=12)
    tubelet_size: Optional[int] = Field(description="", default=1)
    tile_size: Optional[int] = None
    head_channel_list: Optional[list] = [256]
    pretrained_weights_path: Optional[str] = None
    pretrained_model_name: str = None


class TuneTemplateParameters(
    TemplateUserDefinedParams, ModelDatasetParams, ModelBaseParams
):
    """
    A class to define tuning parameters for model training and evaluation.

    Attributes
    ----------
    tune_id : Optional[str]
        The tune id. Defaults to an empty string.
    data_root : Optional[str]
        The root directory for data. Defaults to an empty string.
    mount_root : Optional[str]
        The root directory for mounting data. Defaults to an empty string.
    backbone_model_root: Optional[str]
        The base model root.
    data_id : Optional[str]
        The dataset id. Defaults to an empty string.
    all_data_dir : Optional[str]
        The directory containing all data. Defaults to an empty string.
    all_labels_dir : Optional[str]
        The directory containing all labels. Defaults to an empty string.
    mlflow_tracking_url : Optional[str]
        The mlflow tracking url of the deployed server.
    mlflow_tags Optional[dict] :
        The unique ids(tags) to track an mlflow experiment.
    depth : Optional[int]
        The depth of the model, initialized from `num_layers` in the base models if not set.

    Examples
    --------
    >>> params = TuneTemplateParameters(
    ...     tune_id="tune1",
    ...     data_root="/path/to/data",
    ...     mount_root="/mnt/data",
    ... )
    >>> print(params)
    TuneTemplateParameters(tune_id='tune1', data_root='/path/to/data', mount_root='/mnt/data',
                           data_id='', mmseg_path=None, all_data_dir='',
                           all_labels_dir='', gfm_ckpt='')

    """

    tune_id: Optional[str] = Field(default="", description="")
    data_root: Optional[str] = Field(
        default="", description="Path in the pod where the files PVC is mounted"
    )
    mount_root: Optional[str] = Field(default="", description="")
    backbone_model_root: Optional[str] = Field(default="", description="")
    data_id: Optional[str] = Field(default="", description="")
    all_data_dir: Optional[str] = Field(default="", description="")
    all_labels_dir: Optional[str] = Field(default="", description="")
    mlflow_tracking_url: Optional[str] = Field(
        description="MlFlow url to track fine-tuning metrics",
        default=settings.MLFLOW_URL,
    )
    mlflow_tags: Optional[dict] = Field(
        default=dict,
        description=(
            "Mlflow tags to uniquely match an experiment with unique"
            "identifiers e.g user email address, name ..."
        ),
    )

    # Fields `num_layers` from the BaseModels
    depth: Optional[int] = None

    @model_validator(mode="after")
    def update_fields_before_validation(cls, values):
        values.data.workers_per_gpu = values.data.workers_per_gpu or values.num_workers
        values.data.batch_size = values.data.batch_size or values.batch_size
        values.depth = values.depth or values.num_layers

        # Dataset values to override
        values.ignore_index = values.data.ignore_index or values.ignore_index
        values.constant_multiply = (
            values.data.constant_multiply or values.constant_multiply
        )
        values.classes = values.data.classes or values.classes
        values.class_weights = values.data.class_weights or values.class_weights
        values.norm_means = values.data.norm_means or values.norm_means
        values.norm_stds = values.data.norm_stds or values.norm_stds
        values.output_bands = values.data.tuning_bands or values.output_bands

        return values

    @model_validator(mode="before")
    def update_fields_after_validation(cls, values):

        data = values.get("data", {})
        model = values.get("model", {})

        try:
            if "bands" in data:
                data["bands"] = data.get("bands") or values["bands"]
        except KeyError as exc:
            logger.exception("Error constructing dataset.")
            detail = f"Dataset parameter config error. Missing: {str(exc)}"
            raise HTTPException(status_code=500, detail=detail)
        if model:
            loss_type = (
                values.get("model", {})
                .get("decode_head", {})
                .get("loss_decode", {})
                .get("type")
            )
            if mapped_loss_type := getattr(LossFunctionType, loss_type, loss_type):
                if isinstance(mapped_loss_type, str):
                    values["model"]["decode_head"]["loss_decode"][
                        "type"
                    ] = mapped_loss_type
                else:
                    values["model"]["decode_head"]["loss_decode"][
                        "type"
                    ] = mapped_loss_type.value

        values.update(data)
        return values
