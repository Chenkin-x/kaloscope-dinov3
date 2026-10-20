import math

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

CHECKPOINT_VERSION = 3


class Encoder(nn.Module):
    """Uint8 RGB encoder; eval returns pooled backbone and projector features."""

    def __init__(self, config, initialize=True):
        super().__init__()
        if initialize and config.get("name_or_path"):
            self.backbone = AutoModel.from_pretrained(config["name_or_path"])
        else:
            # Resume reconstructs the backbone from the config saved in the checkpoint.
            architecture = dict(config["backbone_config"])
            model_type = architecture.pop("model_type")
            self.backbone = AutoModel.from_config(AutoConfig.for_model(model_type, **architecture))
        self.model_config = {**config, "backbone_config": self.backbone.config.to_dict()}
        self.pooling = config.get("pooling", "cls")
        if self.pooling not in ("cls", "mean", "cls_mean"):
            raise ValueError("pooling must be cls, mean or cls_mean")
        width = self.backbone.config.hidden_size
        self.feature_dim = width * (2 if self.pooling == "cls_mean" else 1)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, config.get("projection_dim", 128)),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.log_temperature = torch.nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def set_train_from_block(self, first):
        if self.backbone.config.model_type != "dinov3_vit":
            raise ValueError("Temporal training requires a DINOv3 ViT")
        blocks = self.backbone.model.layer
        if not 0 <= first < len(blocks):
            raise ValueError("train_from_block must be a valid zero-based block index")
        self.backbone.requires_grad_(False)
        for block in blocks[first:]:
            block.requires_grad_(True)
        self.backbone.norm.requires_grad_(True)

    def forward(self, images):
        pixels = (images.float() / 255 - self.mean) / self.std
        # DeepSpeed casts the backbone weights; uint8 preprocessing still runs in float32.
        pixels = pixels.to(dtype=next(self.backbone.parameters()).dtype)
        tokens = self.backbone(pixel_values=pixels, return_dict=True).last_hidden_state
        cls = tokens[:, 0]
        patches = tokens[:, 1 + self.backbone.config.num_register_tokens:]
        features = cls if self.pooling == "cls" else patches.mean(1)
        if self.pooling == "cls_mean":
            features = torch.cat((cls, patches.mean(1)), dim=1)
        projections = self.projector(features)
        if self.training:
            return projections, self.log_temperature.exp().clamp(max=100), self.bias
        return torch.cat([features, projections], dim=1)
