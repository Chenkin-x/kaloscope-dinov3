import math

import torch

from dinov3.finetune.model import FineTuneModel


class Encoder(FineTuneModel):
    """Uint8 RGB encoder; eval returns pooled backbone and projector features."""

    def __init__(self, config, initialize=True):
        super().__init__(config, num_classes=0, initialize=initialize)
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
        features = super().forward(pixels)["features"]
        projections = self.projector(features)
        if self.training:
            return projections, self.log_temperature.exp().clamp(max=100), self.bias
        return torch.cat([features, projections], dim=1)
