import math
import torch

OPTIMIZATION_DEFAULTS = {
    "scheduler": "constant",
    "schedule_epochs": None,
    "warmup_epochs": 0.0,
    "warmup_start_lr": 0.0,
    "min_lr": 0.0,
    "weight_decay_end": None,
    "layer_decay": 1.0,
    "backbone_lr_multiplier": 1.0,
    "head_lr_multiplier": 1.0,
    "projector_lr_multiplier": 1.0,
    "patch_embed_lr_multiplier": 1.0,
    "exclude_bias_norm_decay": False,
    "freeze_last_layer_epochs": 0.0,
    "label_smoothing": 0.0,
    "ema_decay": None,
    "ema_decay_end": None,
}


def validate_optimization(config):
    options = config["optimization"]
    unknown = set(options) - set(OPTIMIZATION_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown optimization options: {sorted(unknown)}")
    if options["scheduler"] not in ("constant", "cosine"):
        raise ValueError("optimization.scheduler must be constant or cosine")
    if not 0 < options["layer_decay"] <= 1:
        raise ValueError("layer_decay must be in (0, 1]")
    for key in ("backbone_lr_multiplier", "head_lr_multiplier", "projector_lr_multiplier", "patch_embed_lr_multiplier"):
        if not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("warmup_epochs", "freeze_last_layer_epochs", "warmup_start_lr", "min_lr"):
        if not math.isfinite(options[key]) or options[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if max(options["min_lr"], options["warmup_start_lr"]) > config["lr"]:
        raise ValueError("min_lr and warmup_start_lr must not exceed lr")
    if options["schedule_epochs"] is not None and options["schedule_epochs"] <= 0:
        raise ValueError("schedule_epochs must be positive")
    if not 0 <= options["label_smoothing"] < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    for value in (config["weight_decay"], options["weight_decay_end"]):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("Weight decay values must be finite and nonnegative")
    for key in ("ema_decay", "ema_decay_end"):
        if options[key] is not None and not 0 <= options[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if options["ema_decay_end"] is not None and options["ema_decay"] is None:
        raise ValueError("ema_decay_end requires ema_decay")


def backbone_layer(name, backbone):
    if backbone.config.model_type == "dinov3_vit":
        last_layer = backbone.config.num_hidden_layers + 1
        if name.startswith("model.layer."):
            return int(name.split(".")[2]) + 1, last_layer
        if name.startswith(("embeddings.", "rope_embeddings.")):
            return 0, last_layer
        return last_layer, last_layer
    if backbone.config.model_type == "dinov3_convnext":
        depths = backbone.config.depths
        last_layer = sum(depths) + 1
        if name.startswith("model.stages."):
            parts = name.split(".")
            stage = int(parts[2])
            offset = sum(depths[:stage])
            layer = offset + int(parts[4]) + 1 if parts[3] == "layers" else offset
            return layer, last_layer
        return last_layer, last_layer
    raise ValueError("Layer-wise learning rates require a ViT or ConvNeXt backbone")


def parameter_groups(model, config):
    options = config["optimization"]
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        section, relative_name = name.split(".", 1)
        multiplier = options[f"{section}_lr_multiplier"]
        if section == "backbone":
            layer, last_layer = backbone_layer(relative_name, model.backbone)
            multiplier *= options["layer_decay"] ** (last_layer - layer)
            if relative_name.startswith(("embeddings.patch_embeddings.", "model.stages.0.downsample_layers.")):
                multiplier *= options["patch_embed_lr_multiplier"]
        no_decay = parameter.ndim <= 1 or any(
            token in name for token in ("norm", "gamma", "token", "pos_embed", "fourier_w")
        )
        decay_multiplier = 0.0 if options["exclude_bias_norm_decay"] and no_decay else 1.0
        is_last_layer = section == "head" or name.startswith(f"projector.{len(model.projector) - 1}.")
        key = (multiplier, decay_multiplier, is_last_layer)
        if key not in groups:
            groups[key] = {
                "params": [],
                "parameter_names": [],
                "lr_multiplier": multiplier,
                "wd_multiplier": decay_multiplier,
                "is_last_layer": is_last_layer,
            }
        groups[key]["params"].append(parameter)
        groups[key]["parameter_names"].append(name)
    return list(groups.values())


def cosine_value(start, end, step, steps):
    progress = min(max(step / max(steps - 1, 1), 0.0), 1.0)
    return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2


class TrainingSchedule:
    def __init__(self, config, steps_per_epoch):
        self.config = config
        options = config["optimization"]
        self.total_steps = max(1, math.ceil((options["schedule_epochs"] or config["epochs"]) * steps_per_epoch))
        self.warmup_steps = math.ceil(options["warmup_epochs"] * steps_per_epoch)
        self.freeze_steps = math.ceil(options["freeze_last_layer_epochs"] * steps_per_epoch)
        self.steps_per_epoch = steps_per_epoch
        self.step = 0
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_epochs must be shorter than the scheduling horizon")

    def apply(self, optimizer):
        options = self.config["optimization"]
        base_lr = self.config["lr"]
        if self.step < self.warmup_steps:
            fraction = self.step / self.warmup_steps
            base_lr = options["warmup_start_lr"] + (base_lr - options["warmup_start_lr"]) * fraction
        elif options["scheduler"] == "cosine":
            base_lr = cosine_value(
                base_lr, options["min_lr"], self.step - self.warmup_steps, self.total_steps - self.warmup_steps
            )
        weight_decay = self.config["weight_decay"]
        if options["weight_decay_end"] is not None:
            weight_decay = cosine_value(weight_decay, options["weight_decay_end"], self.step, self.total_steps)
        for group in optimizer.param_groups:
            frozen = group.get("is_last_layer", False) and self.step < self.freeze_steps
            group["lr"] = 0.0 if frozen else base_lr * group.get("lr_multiplier", 1.0)
            group["weight_decay"] = 0.0 if frozen else weight_decay * group.get("wd_multiplier", 1.0)
        return base_lr, weight_decay

    def ema_decay(self):
        options = self.config["optimization"]
        start, end = options["ema_decay"], options["ema_decay_end"]
        return start if end is None else cosine_value(start, end, self.step, self.total_steps)

    def state_dict(self):
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "freeze_steps": self.freeze_steps,
            "steps_per_epoch": self.steps_per_epoch,
        }

    def load_state_dict(self, state):
        if state["steps_per_epoch"] != self.steps_per_epoch:
            raise ValueError("Resume requires unchanged steps per epoch")
        for key in ("step", "total_steps", "warmup_steps", "freeze_steps"):
            setattr(self, key, state[key])


class ModelEMA:
    def __init__(self, state):
        self.updates = 0
        self.shadow = {
            name: value.detach().to(device="cpu", dtype=torch.float32).clone()
            if value.is_floating_point()
            else value.detach().cpu().clone()
            for name, value in state.items()
        }

    @torch.no_grad()
    def update(self, state, decay):
        for name, value in state.items():
            source = value.detach().to(device="cpu", dtype=self.shadow[name].dtype)
            if value.is_floating_point():
                self.shadow[name].lerp_(source, 1 - decay)
            else:
                self.shadow[name].copy_(source)
        self.updates += 1

    def state_dict(self):
        return {"updates": self.updates, "model": self.shadow}

    def load_state_dict(self, state):
        if self.shadow.keys() != state["model"].keys():
            raise ValueError("EMA checkpoint architecture differs from model")
        for name in self.shadow:
            self.shadow[name].copy_(state["model"][name])
        self.updates = state["updates"]
