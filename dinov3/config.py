from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf

DEFAULTS = {
    "model": {
        "name_or_path": "facebook/dinov3-vitb16-pretrain-lvd1689m", "pooling": "cls_mean", "projection_dim": 256
    },
    "paths": None,
    "metadata": None,
    "panel": None,
    "output": "runs/temporal",
    "batch_size": 256,
    "image_size": 512,
    "workers": 16,
    "eval_workers": 8,
    "eval_batch_size": 128,
    "precision": "bf16",
    "seed": 20261004,
    "window_days": 365,
    "train_from_block": 8,
    "backbone_lr": 2e-5,
    "projector_lr": 2e-4,
    "temperature_bias_lr": 1e-3,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "warmup_steps": 1000,
    "eval_every": 1000,
    "save_every": 500,
    "log_every": 20,
    "max_coverage_passes": 50,
    "min_effective_rank": 16.0,
    "min_feature_std": 1e-4,
    "early_stop": {
        "min_delta": 0.001,
        "reduce_patience": 3,
        "stop_patience": 6,
        "min_factor": 0.0625,
        "min_coverage": 2.0,
        "regression_tolerance": 0.05,
        "failure_patience": 3,
    },
}


def resolve_config(path):
    supplied = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown temporal training configuration keys")
    config = {**deepcopy(DEFAULTS), **supplied}
    for key in ("model", "early_stop"):
        config[key] = {**DEFAULTS[key], **supplied.get(key, {})}
    for key in ("paths", "metadata", "panel", "output"):
        if not config[key]:
            raise ValueError(f"Set {key} in the configuration")
        config[key] = str(Path(config[key]).expanduser().resolve())
    validate_config(config)
    return config


def validate_config(config):
    for key in (
        "batch_size",
        "image_size",
        "eval_batch_size",
        "eval_every",
        "save_every",
        "log_every",
        "max_coverage_passes",
    ):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("workers", "eval_workers", "window_days", "warmup_steps", "train_from_block", "seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["precision"] not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    for key in ("backbone_lr", "projector_lr", "temperature_bias_lr", "gradient_clip"):
        if not 0 < config[key] < float("inf"):
            raise ValueError(f"{key} must be finite and positive")
    for key in ("weight_decay", "min_effective_rank", "min_feature_std"):
        if not 0 <= config[key] < float("inf"):
            raise ValueError(f"{key} must be finite and nonnegative")
    stop = config["early_stop"]
    if set(stop) != set(DEFAULTS["early_stop"]):
        raise ValueError("Unknown early_stop settings")
    for key in ("reduce_patience", "stop_patience", "failure_patience"):
        if type(stop[key]) is not int or stop[key] < 1:
            raise ValueError(f"early_stop.{key} must be a positive integer")
    if not 0 < stop["min_factor"] <= 1:
        raise ValueError("early_stop.min_factor must be in (0, 1]")
    for key in ("min_delta", "min_coverage", "regression_tolerance"):
        if not 0 <= stop[key] < float("inf"):
            raise ValueError(f"early_stop.{key} must be finite and nonnegative")
