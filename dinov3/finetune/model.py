"""Transformers backbones with the project's pooling and task heads."""
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

CHECKPOINT_VERSION = 3


def read_config(path):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def load_weights(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def build_backbone(config, initialize=True):
    """Load a Hub/local model, or construct from an explicit saved HF config."""
    unknown = set(config) - {"name_or_path", "backbone_config", "pooling", "projection_dim"}
    if unknown:
        raise ValueError(f"Unknown model options {sorted(unknown)}; use name_or_path or backbone_config")
    if initialize and config.get("name_or_path"):
        backbone = AutoModel.from_pretrained(config["name_or_path"])
    else:
        # Checkpoints carry this config, so resume never fetches initial weights.
        architecture = dict(config["backbone_config"])
        model_type = architecture.pop("model_type")
        backbone = AutoModel.from_config(AutoConfig.for_model(model_type, **architecture))
    if backbone.config.model_type not in ("dinov3_vit", "dinov3_convnext"):
        raise ValueError("Use a Transformers DINOv3 ViT or ConvNeXt model")
    return backbone


class FineTuneModel(nn.Module):
    def __init__(self, config, num_classes=0, initialize=True):
        super().__init__()
        self.backbone = build_backbone(config, initialize)
        self.model_config = {**config, "backbone_config": self.backbone.config.to_dict()}
        self.pooling = config.get("pooling", "cls")
        if self.pooling not in ("cls", "mean", "cls_mean"):
            raise ValueError("pooling must be cls, mean or cls_mean")
        architecture = self.backbone.config
        width = architecture.hidden_size if architecture.model_type == "dinov3_vit" else architecture.hidden_sizes[-1]
        self.feature_dim = width * (2 if self.pooling == "cls_mean" else 1)
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
        tokens = self.backbone(pixel_values=images, return_dict=True).last_hidden_state
        cls = tokens[:, 0]
        # ViT returns [CLS, registers, patches]; ConvNeXt returns [pooled, patches].
        patch = tokens[:, 1 + getattr(self.backbone.config, "num_register_tokens", 0):]
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
