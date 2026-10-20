"""Tiny deterministic fixtures for CPU tests, never used by production training."""

import json
from copy import deepcopy

import numpy as np
from PIL import Image

from dinov3.config import DEFAULTS


def make_fixture(root):
    data = root / "data"
    data.mkdir()
    paths, artists, splits = [], [], []
    for artist in range(2):
        for row in range(8):
            index = len(paths)
            filename = f"{index}.png"
            pixels = np.random.default_rng(index).integers(0, 256, (40, 56, 3), dtype="uint8")
            Image.fromarray(pixels).save(data / filename)
            paths.append(filename)
            artists.append(artist)
            splits.append(0 if row < 6 else (1 if row == 6 else 2))
    np.save(data / "paths.npy", np.array(paths))
    arrays = {
        "artist_ids": artists,
        "days": np.arange(16) * 10,
        "work_family_id": np.arange(16),
        "group_ids": artists,
        "split": splits,
    }
    for key, value in arrays.items():
        np.save(data / f"{key}.npy", np.array(value, dtype="uint8" if key == "split" else "int64"))
    gallery = np.flatnonzero(np.array(splits) == 0)
    query = np.flatnonzero(np.array(splits) == 1)
    np.save(data / "gallery.npy", gallery)
    np.save(data / "query.npy", query)
    content = {str(int(i)): ["gallery_character", "gallery_copyright"] for i in gallery}
    content.update({str(int(i)): ["query_character", "query_copyright"] for i in query})
    (data / "content.json").write_text(json.dumps(content))
    config = deepcopy(DEFAULTS)
    config.update(
        paths=str(data / "paths.npy"),
        metadata=str(data),
        panel=str(data),
        output=str(root / "run"),
        model={
            "name_or_path": None,
            "pooling": "cls_mean",
            "projection_dim": 16,
            "backbone_config": {
                "model_type": "dinov3_vit",
                "image_size": 32,
                "patch_size": 16,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 64,
                "num_register_tokens": 4,
            },
        },
        batch_size=8,
        image_size=32,
        workers=0,
        eval_workers=0,
        eval_batch_size=4,
        precision="fp32",
        train_from_block=1,
        warmup_steps=1,
        eval_every=2,
        save_every=1,
        min_effective_rank=0.0,
        min_feature_std=0.0,
        max_coverage_passes=2,
    )
    return config
