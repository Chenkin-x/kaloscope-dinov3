from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


def prepare_rgb(image):
    """Apply orientation and composite transparency before discarding alpha."""
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def rgb_loader(path):
    with Image.open(path) as image:
        return prepare_rgb(image)


class Originals(Dataset):
    """Index-aligned path manifest; short-side resize and center crop, no stretching."""

    def __init__(self, paths, image_size=512):
        self.manifest = Path(paths).resolve()
        self.paths = np.load(self.manifest, mmap_mode="r", allow_pickle=False)
        if self.paths.ndim != 1 or self.paths.dtype.kind not in "SU":
            raise ValueError("paths.npy must be a one-dimensional fixed-width byte/unicode array")
        self.spatial = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        path = Path(path.decode("utf-8") if isinstance(path, bytes) else str(path))
        if not path.is_absolute():
            path = self.manifest.parent / path
        image = self.spatial(rgb_loader(path))
        return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1), int(index)


def load_metadata(root, count):
    arrays = {}
    for name in ("artist_ids", "days", "work_family_id", "group_ids", "split"):
        array = np.load(Path(root) / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != (count,) or array.dtype.kind not in "iu":
            raise ValueError(f"{name}.npy must be an integer array aligned with paths.npy")
        if name == "days" and array.dtype.kind != "i":
            raise ValueError("days.npy must use signed integers for temporal differences")
        if array.dtype == np.dtype("uint64"):
            if (array > np.iinfo(np.int64).max).any():
                raise ValueError(f"{name}.npy IDs must fit in signed int64")
            array = array.astype("int64")
        if name == "days":
            array = array.astype("int64", copy=False)
        arrays[name] = array
    if not np.isin(arrays["split"], [0, 1, 2]).all():
        raise ValueError("split codes must be 0=train, 1=dev, 2=test")
    assigned = arrays["group_ids"] >= 0
    if (arrays["artist_ids"][assigned] < 0).any() or (arrays["work_family_id"][assigned] < 0).any():
        raise ValueError("Assigned images need nonnegative single-artist and work-family IDs")
    grouped = np.flatnonzero(assigned)
    grouped = grouped[np.argsort(arrays["group_ids"][grouped], kind="stable")]
    if (
        (arrays["group_ids"][grouped][1:] == arrays["group_ids"][grouped][:-1])
        & (arrays["artist_ids"][grouped][1:] != arrays["artist_ids"][grouped][:-1])
    ).any():
        raise ValueError("Each year group must belong to exactly one artist")
    # A work family may have multiple rows, but must never cross split boundaries.
    families = arrays["work_family_id"]
    order = np.argsort(families, kind="stable")
    if (
        (families[order][1:] == families[order][:-1])
        & (arrays["split"][order][1:] != arrays["split"][order][:-1])
        & (families[order][1:] >= 0)
    ).any():
        raise ValueError("Work-family leakage across train/dev/test splits")
    return arrays
