from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


def create_accelerator(config, gradient_clip):
    accelerator = Accelerator(
        mixed_precision="no" if config["precision"] == "fp32" else config["precision"],
        gradient_accumulation_steps=1,
        # DINOv3's mask token is unused by image fine-tuning (also covers explicit CPU/DDP checks).
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    plugin = accelerator.state.deepspeed_plugin
    if plugin is not None:
        # Both loaders already partition samples by rank; do not prepare/shard them again.
        plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = config["batch_size"]
        plugin.deepspeed_config["gradient_clipping"] = gradient_clip
    return accelerator
