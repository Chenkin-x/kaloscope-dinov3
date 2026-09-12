import inspect
from pathlib import Path

import torch
from torch import nn

from dinov3.hub import backbones


MODEL_NAMES = tuple(name for name in vars(backbones) if name.startswith("dinov3_"))


def read_config(path):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def load_weights(path):
    if str(path).startswith("https://"):
        return torch.hub.load_state_dict_from_url(str(path), map_location="cpu", weights_only=True)
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def state_dict_from(payload):
    for key in ("model", "state_dict", "teacher", "student"):
        if key in payload and isinstance(payload[key], dict):
            return state_dict_from(payload[key])
    return {key.removeprefix("module."): value for key, value in payload.items()}


def build_backbone(config, initialize=True):
    name = config.get("name", "dinov3_vits16")
    kwargs = config.get("kwargs", {})
    weights = config.get("weights") if initialize else None
    if name in ("custom_vit", "custom_convnext"):
        if name == "custom_vit":
            from dinov3.models.vision_transformer import DinoVisionTransformer

            factory = DinoVisionTransformer
        else:
            from dinov3.models.convnext import ConvNeXt

            factory = ConvNeXt
        unknown = set(kwargs) - set(inspect.signature(factory).parameters)
        if unknown:
            raise ValueError(f"Unknown custom architecture options: {sorted(unknown)}")
        backbone = factory(**kwargs)
        if name == "custom_vit":
            backbone.init_weights()
    else:
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model {name}; available: {MODEL_NAMES}, custom_vit, custom_convnext")
        if weights in backbones.Weights.__members__:
            return getattr(backbones, name)(pretrained=True, weights=backbones.Weights[weights], **kwargs)
        backbone = getattr(backbones, name)(pretrained=False, **kwargs)
    if weights:
        state = state_dict_from(load_weights(weights))
        if any(key.startswith("backbone.") for key in state):
            state = {
                key.removeprefix("backbone."): value for key, value in state.items() if key.startswith("backbone.")
            }
        backbone.load_state_dict(state, strict=True)
    return backbone


class FineTuneModel(nn.Module):
    def __init__(self, config, num_classes=0, initialize=True):
        super().__init__()
        self.backbone = build_backbone(config, initialize)
        self.pooling = config.get("pooling", "cls")
        if self.pooling not in ("cls", "mean", "cls_mean"):
            raise ValueError("pooling must be cls, mean or cls_mean")
        self.feature_dim = self.backbone.embed_dim * (2 if self.pooling == "cls_mean" else 1)
        self.head = nn.Linear(self.feature_dim, num_classes) if num_classes else None
        projection_dim = config.get("projection_dim", 128)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim), nn.GELU(), nn.Linear(self.feature_dim, projection_dim)
        )
        self.frozen_backbone = False

    def set_train_scope(self, scope, contrastive):
        if scope not in ("backbone", "head", "all"):
            raise ValueError("train_scope must be backbone, head or all")
        if scope == "head" and self.head is None:
            raise ValueError("Head-only training requires a classifier")
        self.frozen_backbone = scope == "head"
        self.backbone.requires_grad_(not self.frozen_backbone)
        if self.head is not None:
            self.head.requires_grad_(scope != "backbone")
        self.projector.requires_grad_(contrastive and scope != "head")

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_backbone:
            self.backbone.eval()
        return self

    def forward(self, images, contrastive=False, patches=False):
        tokens = self.backbone(images, is_training=True)
        cls = tokens["x_norm_clstoken"]
        patch = tokens["x_norm_patchtokens"]
        features = cls if self.pooling == "cls" else patch.mean(1)
        if self.pooling == "cls_mean":
            features = torch.cat((cls, patch.mean(1)), dim=1)
        result = {"features": features}
        if self.head is not None:
            result["logits"] = self.head(features)
        if contrastive:
            result["projections"] = self.projector(features)
        if patches:
            result["patch_tokens"] = patch
        return result
