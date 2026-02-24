# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import shutil
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Literal

import torch
import torchvision
import torchvision.transforms.v2 as v2
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

from solo.data.custom.imagenet import ImgNetDataset_42
from foveation.factory import FoveationTransform, build_foveation
from solo.data.custom.base import H5ClassificationDataset
from solo.data.custom.core50 import Core50, Core50ForBGClassification

try:
    from solo.data.h5_dataset import H5Dataset
except ImportError:
    _h5_available = False
else:
    _h5_available = True


def build_pre_transform(
        mode: Literal['rrc', 'cc', 'r', 'rcc'],
        size: Union[int, Tuple[int, int]],
        cc_size: int = None,
        scale: Tuple[float, float] = (0.08, 1.0),
) -> Callable:
    """Transform to prepare an image for GPU batch processing (spatial ops + ToImage).

    Returns a uint8 image tensor. Augmentation and normalisation are applied
    separately on the GPU via the companion T_train / T_val transforms.

    Args:
        mode: Spatial operation to apply.
            ``'r'``   – Resize to ``size``.
            ``'cc'``  – CenterCrop to ``cc_size``.
            ``'rcc'`` – Resize to ``size``, then CenterCrop to ``cc_size``.
            ``'rrc'`` – RandomResizedCrop to ``size`` with ``scale``.
        size: Target size (int → shorter-side resize; tuple → exact HxW).
        cc_size: Crop size used with ``'cc'`` and ``'rcc'`` modes.
        scale: Scale range for RandomResizedCrop (only used with ``'rrc'``).
    """
    pipeline = []

    if mode == "r":
        pipeline.append(v2.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True))
    elif mode == "cc":
        pipeline.append(v2.CenterCrop(cc_size))
    elif mode == "rcc":
        pipeline.append(v2.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True))
        pipeline.append(v2.CenterCrop(cc_size))
    elif mode == "rrc":
        pipeline.append(v2.RandomResizedCrop(
            size, scale=scale, interpolation=InterpolationMode.BICUBIC, antialias=True,
        ))
    else:
        raise ValueError(f"Unknown mode: {mode}, expected one of ['rrc', 'cc', 'r', 'rcc']")

    pipeline.append(v2.ToImage())
    return v2.Compose(pipeline)


def build_custom_pipeline() -> dict:
    """Split-format pipeline for custom ImageFolder data.

    ``T_Pre_*`` run in the dataset (CPU); ``T_train`` / ``T_val`` run on the GPU.
    """
    return {
        "T_Pre_Train": build_pre_transform('rrc', size=224),
        "T_train": v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]),
        "T_Pre_Val": build_pre_transform('rcc', size=256, cc_size=224),
        "T_val": v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]),
    }


def _get_pipeline(dataset: str) -> dict:
    """Return the split pipeline dict for *dataset*.

    Each dict contains four keys:

    * ``T_Pre_Train`` – spatial CPU transform for training (→ uint8 tensor).
    * ``T_train``     – augmentation + normalisation to run on the GPU.
    * ``T_Pre_Val``   – spatial CPU transform for validation (→ uint8 tensor).
    * ``T_val``       – normalisation to run on the GPU.
    """

    # ------------------------------------------------------------------ CIFAR
    cifar_pipeline = {
        "T_Pre_Train": build_pre_transform('rrc', size=32),
        "T_train": v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]),
        # Images are already 32×32; only PIL→tensor conversion needed.
        "T_Pre_Val": v2.ToImage(),
        "T_val": v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]),
    }

    cifar_pipeline_224 = {
        "T_Pre_Train": build_pre_transform('r', size=224),
        "T_train": v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]),
        "T_Pre_Val": build_pre_transform('rcc', size=224, cc_size=224),
        "T_val": v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]),
    }

    # ------------------------------------------------------------------- STL
    stl_pipeline = {
        "T_Pre_Train": build_pre_transform('rrc', size=96),
        "T_train": v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]),
        "T_Pre_Val": build_pre_transform('r', size=(96, 96)),
        "T_val": v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]),
    }

    stl_pipeline_224 = {
        "T_Pre_Train": build_pre_transform('rrc', size=224),
        "T_train": v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]),
        "T_Pre_Val": build_pre_transform('r', size=(224, 224)),
        "T_val": v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]),
    }

    # --------------------------------------------------------------- ImageNet
    imagenet_pipeline = {
        "T_Pre_Train": build_pre_transform('rrc', size=224),
        "T_train": v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]),
        # Standard: resize shorter side to 256, centre-crop to 224.
        "T_Pre_Val": build_pre_transform('rcc', size=256, cc_size=224),
        "T_val": v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]),
    }

    # ---------------------------------------------------------------- Toybox
    # Larger crop scale (0.6) because objects already fill most of the frame.
    toybox_pipeline = {
        "T_Pre_Train": build_pre_transform('rrc', size=224, scale=(0.6, 1.0)),
        "T_train": v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]),
        "T_Pre_Val": build_pre_transform('rcc', size=256, cc_size=224),
        "T_val": v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]),
    }

    # ---------------------------------------------------------------- COIL100
    coil100_pipeline = {
        "T_Pre_Train": build_pre_transform('rrc', size=224),
        "T_train": v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]),
        "T_Pre_Val": build_pre_transform('r', size=224),
        "T_val": v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]),
    }

    custom_pipeline = build_custom_pipeline()

    pipelines = {
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "cifar10_224": cifar_pipeline_224,
        "cifar100_224": cifar_pipeline_224,
        "STL10": stl_pipeline,
        "STL10_224": stl_pipeline_224,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "imagenet_42": imagenet_pipeline,
        "imagenet100_42": imagenet_pipeline,
        "imagenet1pct_42": imagenet_pipeline,
        "imagenet10pct_42": imagenet_pipeline,
        "custom": custom_pipeline,
        "core50": imagenet_pipeline,
        "DTD": imagenet_pipeline,
        "Flowers102": imagenet_pipeline,
        "FGVCAircraft": imagenet_pipeline,
        "Food101": imagenet_pipeline,
        "OxfordIIITPet": imagenet_pipeline,
        "Places365": imagenet_pipeline,
        "COIL100": coil100_pipeline,
        "StanfordCars": imagenet_pipeline,
        "Places365_h5": imagenet_pipeline,
        "SUN397": imagenet_pipeline,
        "SUN397_h5": imagenet_pipeline,
        "toybox": toybox_pipeline,
    }

    assert dataset in pipelines, f"{dataset} is not supported."
    return pipelines[dataset]


def prepare_transforms(dataset: str) -> Tuple[nn.Module, nn.Module]:
    """Return **combined** (pre-transform + GPU transform) pipelines for *dataset*.

    The returned transforms cover the full CPU pipeline and are suitable for
    use-cases that do **not** split CPU pre-processing from GPU augmentation
    (e.g. KNN evaluation, feature extraction).

    For GPU-accelerated augmentation, use :func:`_get_pipeline` directly or
    go through :func:`prepare_data`.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transform pipelines.
    """
    pipeline = _get_pipeline(dataset)
    T_train = v2.Compose([pipeline["T_Pre_Train"], pipeline["T_train"]])
    T_val = v2.Compose([pipeline["T_Pre_Val"], pipeline["T_val"]])
    return T_train, T_val


def prepare_datasets(
        dataset: str,
        T_train: Callable,
        T_val: Callable,
        train_data_path: Optional[Union[str, Path]] = None,
        val_data_path: Optional[Union[str, Path]] = None,
        data_format: Optional[str] = "image_folder",
        download: bool = True,
        data_fraction: float = -1.0,
        foveation_cfg: Optional[dict] = None
) -> Tuple[Dataset, Dataset]:
    """Prepare train and val datasets.

    ``T_train`` / ``T_val`` are applied inside the dataset (CPU).  When using
    GPU augmentation (via :func:`prepare_data`), pass the pre-transforms here
    and apply the GPU transforms via ``on_after_batch_transfer``.

    Args:
        dataset (str): dataset name.
        T_train (Callable): transform applied to each training sample.
        T_val (Callable): transform applied to each validation sample.
        train_data_path: path to training data. Defaults to ``<repo>/datasets``.
        val_data_path: path to validation data. Defaults to ``<repo>/datasets``.
        data_format: ``"image_folder"`` or ``"h5"``. Defaults to ``"image_folder"``.
        download (bool): download the dataset if not present. Defaults to ``True``.
        data_fraction (float): fraction of training data to use (``-1`` = all).
        foveation_cfg: optional foveation configuration dict.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if foveation_cfg is not None:
        fov_type = foveation_cfg.get("type", None)

        if fov_type in ["blur", "cm"]:

            params = foveation_cfg.get(fov_type, {})

            print(
                f"[Foveation] Classification mode: type={fov_type} | "
                + ", ".join(f"{k}={v}" for k, v in params.items())
            )

            foveation = build_foveation(foveation_cfg)

            T_train = FoveationTransform(foveation, T_train)
            T_val = FoveationTransform(foveation, T_val)

        else:
            print("[Foveation] Disabled for classification")


    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if val_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        val_data_path = sandbox_folder / "datasets"

    assert dataset in [
        "cifar10", "cifar100", "stl10", "imagenet", "imagenet100", "custom",
        "imagenet100_42", "imagenet1pct_42", "imagenet10pct_42", "imagenet_42",
        "core50", "DTD", "Flowers102", "FGVCAircraft", "Food101", "OxfordIIITPet",
        "Places365", "StanfordCars", "STL10", "STL10_224", "Places365_h5",
        "SUN397", "Caltech101", "toybox", "COIL100", "Places365_h5", "SUN397_h5",
        "cifar100_224", "cifar10_224",
    ]

    if dataset in ["cifar10", "cifar100", "cifar10_224", "cifar100_224"]:
        if dataset == "cifar10_224":
            dataset = "cifar10"
        if dataset == "cifar100_224":
            dataset = "cifar100"
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            train_data_path, train=True, download=download, transform=T_train,
        )
        val_dataset = DatasetClass(
            val_data_path, train=False, download=download, transform=T_val,
        )

    elif dataset in [
        "DTD", "Flowers102", "FGVCAircraft", "Food101", "OxfordIIITPet",
        "Places365", "StanfordCars", "STL10", "SUN397", "Caltech101", "STL10_224",
    ]:
        if dataset == "STL10_224":
            dataset = "STL10"
        DatasetClass = vars(torchvision.datasets)[dataset]

        if dataset == "StanfordCars" and download:
            download = False
            if not (Path(train_data_path) / "stanford_cars").exists():
                import kagglehub
                kagglehub.login()
                path = kagglehub.dataset_download(
                    "rickyyyyyyy/torchvision-stanford-cars", force_download=False
                )
                ds_path = Path(path) / "stanford_cars"
                shutil.move(ds_path, train_data_path)

        if dataset == "Places365":
            train_split = "train-standard"
        elif dataset == "OxfordIIITPet":
            train_split = "trainval"
        else:
            train_split = "train"

        train_dataset = DatasetClass(
            train_data_path, split=train_split, download=download, transform=T_train,
        )
        val_dataset = DatasetClass(
            val_data_path, split="test", download=download, transform=T_val,
        )

    elif dataset == "Places365_h5":
        train_dataset = H5ClassificationDataset(
            root=Path(train_data_path) / "Places365", transform=T_train, split="train"
        )
        val_dataset = H5ClassificationDataset(
            root=Path(val_data_path) / "Places365", transform=T_val, split="val"
        )

    elif dataset == "SUN397_h5":
        train_dataset = H5ClassificationDataset(
            root=Path(train_data_path) / "SUN397", transform=T_train, split="train"
        )
        val_dataset = H5ClassificationDataset(
            root=Path(val_data_path) / "SUN397", transform=T_val, split="test"
        )

    elif dataset == "COIL100":
        train_dataset = H5ClassificationDataset(
            root=Path(train_data_path) / "coil100", transform=T_train, split="train"
        )
        val_dataset = H5ClassificationDataset(
            root=Path(val_data_path) / "coil100", transform=T_val, split="val"
        )

    elif dataset in ["imagenet", "imagenet100", "custom"]:
        if data_format == "h5":
            assert _h5_available
            train_dataset = H5Dataset(dataset, train_data_path, T_train)
            val_dataset = H5Dataset(dataset, val_data_path, T_val)
        else:
            train_dataset = ImageFolder(train_data_path, T_train)
            val_dataset = ImageFolder(val_data_path, T_val)

    elif dataset in ["imagenet_42", "imagenet100_42", "imagenet1pct_42", "imagenet10pct_42"]:
        if dataset == "imagenet100_42":
            subset = "imgnet100"
        elif dataset == "imagenet1pct_42":
            subset = "1pct"
        elif dataset == "imagenet10pct_42":
            subset = "10pct"
        else:
            subset = None

        train_dataset = ImgNetDataset_42(
            Path(train_data_path) / "ImageNet/h5", T_train, split="train", subset=subset
        )
        val_dataset = ImgNetDataset_42(
            Path(val_data_path) / "ImageNet/h5", T_val, split="val", subset=subset
        )

    elif dataset == "core50":
        train_dataset = Core50(
            h5_path=Path(train_data_path) / "core50_350x350/core50_arr.h5",
            transform=T_train,
            backgrounds=["s1", "s2", "s3", "s4", "s5", "s6"],
        )
        val_dataset = Core50(
            h5_path=Path(val_data_path) / "core50_350x350/core50_arr.h5",
            transform=T_val,
            backgrounds=["s7", "s8", "s9", "s10", "s11"],
        )

    elif dataset == "core50_bg":
        train_dataset = Core50ForBGClassification(
            h5_path=Path(train_data_path) / "core50_350x350/core50_arr.h5",
            split="train", transform=T_train,
        )
        val_dataset = Core50ForBGClassification(
            h5_path=Path(val_data_path) / "core50_350x350/core50_arr.h5",
            split="test", transform=T_val,
        )

    elif dataset == "toybox":
        train_dataset = H5ClassificationDataset(
            Path(train_data_path) / "ToyBox/h5", split="train", transform=T_train
        )
        val_dataset = H5ClassificationDataset(
            Path(val_data_path) / "ToyBox/h5", split="test", transform=T_val
        )

    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        data = train_dataset.samples
        files = [f for f, _ in data]
        labels = [l for _, l in data]

        from sklearn.model_selection import train_test_split

        files, _, labels, _ = train_test_split(
            files, labels, train_size=data_fraction, stratify=labels, random_state=42
        )
        train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset, val_dataset


def prepare_dataloaders(
        train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Wrap datasets in DataLoaders.

    Args:
        train_dataset (Dataset): training data.
        val_dataset (Dataset): validation data.
        batch_size (int): batch size. Defaults to 64.
        num_workers (int): parallel workers. Defaults to 4.

    Returns:
        Tuple[DataLoader, DataLoader]: training and validation dataloaders.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def prepare_data(
        dataset: str,
        train_data_path: Optional[Union[str, Path]] = None,
        val_data_path: Optional[Union[str, Path]] = None,
        data_format: Optional[str] = "image_folder",
        batch_size: int = 64,
        num_workers: int = 4,
        download: bool = True,
        data_fraction: float = -1.0,
        auto_augment: bool = False,
        foveation_cfg: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, nn.Module, nn.Module]:
    """Build dataloaders with split CPU/GPU transforms.

    The spatial pre-transforms (resize/crop + ``ToImage``) run inside the
    :class:`~torch.utils.data.Dataset` on the CPU.  The augmentation and
    normalisation transforms (``T_train_gpu`` / ``T_val_gpu``) are returned
    separately so the caller can apply them on the accelerator via
    ``on_after_batch_transfer``.

    Args:
        dataset (str): dataset name.
        train_data_path: path to training data. Defaults to ``<repo>/datasets``.
        val_data_path: path to validation data. Defaults to ``<repo>/datasets``.
        data_format: ``"image_folder"`` or ``"h5"``. Defaults to ``"image_folder"``.
        batch_size (int): batch size. Defaults to 64.
        num_workers (int): parallel workers. Defaults to 4.
        download (bool): download dataset if missing. Defaults to ``True``.
        data_fraction (float): fraction of training data to use (``-1`` = all).
        auto_augment (bool): use RandAugment via timm (full CPU transform, no
            GPU split). Defaults to ``False``.
        foveation_cfg: optional foveation configuration dict.

    Returns:
        Tuple[DataLoader, DataLoader, nn.Module, nn.Module]:
            training dataloader, validation dataloader,
            GPU training transform, GPU validation transform.
    """
    pipeline = _get_pipeline(dataset)
    T_pre_train: Callable = pipeline["T_Pre_Train"]
    T_train_gpu: nn.Module = pipeline["T_train"]
    T_pre_val: Callable = pipeline["T_Pre_Val"]
    T_val_gpu: nn.Module = pipeline["T_val"]

    # print all the transforms for debugging
    print(f"Dataset: {dataset}")
    print(f"T_pre_train: {T_pre_train}")
    print(f"T_train_gpu: {T_train_gpu}")
    print(f"T_pre_val: {T_pre_val}")
    print(f"T_val_gpu: {T_val_gpu}")

    if auto_augment:
        # timm's create_transform is a complete CPU pipeline (resize + aug +
        # ToTensor + Normalize), so use it as-is in the dataset and skip the
        # GPU split for training.
        T_pre_train = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=None,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        T_train_gpu = nn.Identity()

    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_pre_train,
        T_pre_val,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        data_format=data_format,
        download=download,
        data_fraction=data_fraction,
        foveation_cfg=foveation_cfg,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader, T_train_gpu, T_val_gpu
