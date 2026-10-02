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

    def forward(self, images):
        features = super().forward((images.float() / 255 - self.mean) / self.std)["features"]
        projections = self.projector(features)
        if self.training:
            return projections, self.log_temperature.exp().clamp(max=100), self.bias
        return torch.cat([features, projections], dim=1)
