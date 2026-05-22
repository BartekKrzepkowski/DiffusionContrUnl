import logging
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as torch_transforms
from datasets import load_dataset
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import InterpolationMode

INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}

log = logging.getLogger(__name__)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose(
        [
            torch_transforms.Resize(size, interpolation=interpolation),
            torch_transforms.CenterCrop(size),
            _convert_image_to_rgb,
            torch_transforms.ToTensor(),
            torch_transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transform


class Imagenette(Dataset):
    def __init__(self, split, class_to_forget=None, transform=None):
        self.dataset = load_dataset("frgfm/imagenette", "160px")[split]
        self.class_to_idx = {
            cls: i for i, cls in enumerate(self.dataset.features["label"].names)
        }
        self.file_to_class = {
            str(idx): self.dataset["label"][idx] for idx in range(len(self.dataset))
        }

        self.class_to_forget = class_to_forget
        self.num_classes = max(self.class_to_idx.values()) + 1
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        label = example["label"]

        if example["label"] == self.class_to_forget:
            label = np.random.randint(0, self.num_classes)

        if self.transform:
            image = self.transform(image)
        return image, label


def _load_image_train_dataset(data_path):
    data_path = str(data_path)
    try:
        return load_dataset(data_path)["train"]
    except Exception as first_error:
        path = Path(data_path)
        if path.exists():
            try:
                return load_dataset("imagefolder", data_dir=data_path)["train"]
            except Exception as second_error:
                raise RuntimeError(
                    f"Could not load image dataset from {data_path!r} either as a "
                    "HuggingFace dataset script or as an imagefolder."
                ) from second_error
        raise RuntimeError(
            f"Could not load image dataset from {data_path!r}. "
            "Provide a HuggingFace dataset id/script or an existing image folder."
        ) from first_error


class NSFW(Dataset):
    def __init__(self, data_path="data/nsfw", transform=None):
        self.dataset = _load_image_train_dataset(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if self.transform:
            image = self.transform(image)

        return image


class NOT_NSFW(Dataset):
    def __init__(self, data_path="data/not-nsfw", transform=None):
        self.dataset = _load_image_train_dataset(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if self.transform:
            image = self.transform(image)

        return image


def setup_model(config, ckpt, device):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    # SD v1 checkpoints are trusted local Lightning pickles; PyTorch >=2.6
    # defaults to weights_only=True, which rejects their callback metadata.
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if m:
        log.warning("Missing keys while loading SD checkpoint %s: %s", ckpt, m)
    if u:
        log.warning("Unexpected keys while loading SD checkpoint %s: %s", ckpt, u)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


def setup_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", class_to_forget, transform)
    # train_set = Imagenette('train', transform)

    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def setup_ga_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    class_idx = _coerce_single_class_index(class_to_forget)
    filtered_data = Subset(train_set, _label_indices(train_set, [class_idx], include=True))

    train_dl = DataLoader(filtered_data, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def _as_sequence(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _label_indices(train_set, class_indices, *, include=True):
    class_indices = {int(idx) for idx in _as_sequence(class_indices)}
    if not class_indices:
        raise ValueError("class_indices must not be empty")
    labels = train_set.dataset["label"]
    if include:
        return [idx for idx, label in enumerate(labels) if int(label) in class_indices]
    return [idx for idx, label in enumerate(labels) if int(label) not in class_indices]


def _coerce_single_class_index(class_to_forget):
    values = _as_sequence(class_to_forget)
    if len(values) != 1:
        raise ValueError(f"Expected a single class index, got {class_to_forget!r}")
    value = values[0]
    if isinstance(value, torch.Tensor):
        value = int(value.item())
    return int(value)


def _resolve_forget_class_indices(train_set, class_to_forget=None, forget_classes=None, forget_concepts=None):
    requested = []
    requested.extend(_as_sequence(forget_classes))
    requested.extend(_as_sequence(forget_concepts))
    if not requested:
        requested.extend(_as_sequence(class_to_forget))
    if not requested:
        raise ValueError("No class/concept selected for Stable Diffusion class forgetting.")

    name_to_idx = {str(name).lower().replace("_", " "): idx for name, idx in train_set.class_to_idx.items()}
    indices = set()
    for item in requested:
        if isinstance(item, torch.Tensor):
            item = int(item.item())
        if isinstance(item, int):
            indices.add(int(item))
            continue
        text = str(item).strip()
        if text == "":
            continue
        if text.lstrip("-").isdigit():
            indices.add(int(text))
            continue
        normalized = text.lower().replace("_", " ")
        if normalized not in name_to_idx:
            available = ", ".join(train_set.class_to_idx.keys())
            raise ValueError(f"Unknown forget class/concept '{item}'. Available classes: {available}")
        indices.add(int(name_to_idx[normalized]))

    if not indices:
        raise ValueError("No valid class/concept selected for Stable Diffusion class forgetting.")
    return indices


def setup_class_forgetting_data(
    class_to_forget=None,
    batch_size=8,
    image_size=512,
    interpolation="bicubic",
    forget_classes=None,
    forget_concepts=None,
    retain_shuffle=True,
    forget_shuffle=True,
):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    forget_indices = _resolve_forget_class_indices(
        train_set,
        class_to_forget=class_to_forget,
        forget_classes=forget_classes,
        forget_concepts=forget_concepts,
    )
    retain_indices = _label_indices(train_set, forget_indices, include=False)
    forget_indices_in_dataset = _label_indices(train_set, forget_indices, include=True)
    retain_data = Subset(train_set, retain_indices)
    forget_data = Subset(train_set, forget_indices_in_dataset)
    if not retain_data:
        raise ValueError(f"Retain split is empty for forget classes {sorted(forget_indices)}")
    if not forget_data:
        raise ValueError(f"Forget split is empty for forget classes {sorted(forget_indices)}")

    log.info(
        "SD class forgetting split: retain=%d forget=%d batch_size=%d "
        "retain_shuffle=%s forget_shuffle=%s forget_indices=%s",
        len(retain_data),
        len(forget_data),
        batch_size,
        bool(retain_shuffle),
        bool(forget_shuffle),
        sorted(forget_indices),
    )
    retain_dl = DataLoader(retain_data, batch_size=batch_size, shuffle=bool(retain_shuffle))
    forget_dl = DataLoader(forget_data, batch_size=batch_size, shuffle=bool(forget_shuffle))
    return retain_dl, forget_dl, descriptions, sorted(forget_indices)


def setup_remain_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    class_idx = _coerce_single_class_index(class_to_forget)
    filtered_data = Subset(train_set, _label_indices(train_set, [class_idx], include=False))

    train_dl = DataLoader(filtered_data, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def setup_forget_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    class_idx = _coerce_single_class_index(class_to_forget)
    filtered_data = Subset(train_set, _label_indices(train_set, [class_idx], include=True))
    train_dl = DataLoader(filtered_data, batch_size=batch_size)
    return train_dl, descriptions


def setup_forget_nsfw_data(batch_size, image_size, interpolation="bicubic", nsfw_data_path="data/nsfw", not_nsfw_data_path="data/not-nsfw"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    forget_set = NSFW(data_path=nsfw_data_path, transform=transform)
    forget_dl = DataLoader(forget_set, batch_size=batch_size)

    remain_set = NOT_NSFW(data_path=not_nsfw_data_path, transform=transform)
    remain_dl = DataLoader(remain_set, batch_size=batch_size)
    return forget_dl, remain_dl
