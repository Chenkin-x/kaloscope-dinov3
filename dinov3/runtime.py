from accelerate import Accelerator


def create_accelerator(config, gradient_clip):
    accelerator = Accelerator(
        mixed_precision="no" if config["precision"] == "fp32" else config["precision"],
        gradient_accumulation_steps=1,
    )
    plugin = accelerator.state.deepspeed_plugin
    if plugin is not None:
        # The pair sampler already partitions samples by rank; do not shard the loader again.
        plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = config["batch_size"]
        plugin.deepspeed_config["gradient_clipping"] = gradient_clip
    return accelerator
