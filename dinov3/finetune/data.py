import importlib

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


def _import_callable(dotted_path):
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid transform path: {dotted_path!r} (must be 'module.attr')")
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _build_augment(size, config):
    if not config:
        return []
    ops = []
    if config.get("random_resized_crop", False):
        scale = tuple(config.get("random_resized_crop_scale", (0.5, 1.0)))
        ops.append(transforms.RandomResizedCrop(size, scale=scale))
    if config.get("random_horizontal_flip", False):
        ops.append(transforms.RandomHorizontalFlip())
    if config.get("color_jitter") is not None:
        cj = config["color_jitter"]
        ops.append(transforms.RandomApply(
            [transforms.ColorJitter(
                cj.get("brightness", 0.3),
                cj.get("contrast", 0.3),
                cj.get("saturation", 0.3),
                cj.get("hue", 0.1),
            )],
            p=cj.get("p", 0.8),
        ))
    if config.get("random_grayscale", False):
        ops.append(transforms.RandomGrayscale(p=config.get("random_grayscale_p", 0.2)))
    return ops


def image_transform(size, mean=None, std=None, augment=None, custom_transform=None):
    if custom_transform is not None:
        factory = _import_callable(custom_transform)
        return factory(resize_size=size)
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    spatial = _build_augment(size, augment)
    if not spatial:
        spatial = [transforms.Resize(round(size * 256 / 224)), transforms.CenterCrop(size)]
    return transforms.Compose(spatial + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def lvd_transform(resize_size: int = 256):
    from torchvision.transforms import v2

    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def sat_transform(resize_size: int = 256):
    from torchvision.transforms import v2

    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
    ])


class FolderViews(Dataset):
    def __init__(self, root, transform, two_views=False):
        self.folder = datasets.ImageFolder(root)
        self.transform = transform
        self.two_views = two_views
        self.classes = self.folder.classes

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, index):
        image, label = self.folder[index]
        first = self.transform(image)
        second = self.transform(image) if self.two_views else first
        return first, second, label, index
