import argparse
import json
from pathlib import Path
import random
import time
import uuid

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .contrastive import MemoryBank, contrastive_loss, distributed
from .data import FolderViews, image_transform
from .model import CHECKPOINT_VERSION, FineTuneModel, load_weights, read_config
from .optimization import (
    OPTIMIZATION_DEFAULTS,
    ModelEMA,
    TrainingSchedule,
    parameter_groups,
    validate_optimization,
)
from .runtime import create_accelerator


DEFAULTS = {
    "model": {"name_or_path": "facebook/dinov3-vits16-pretrain-lvd1689m", "projection_dim": 128},
    "data": {"train": "base", "val": None, "image_size": 224, "workers": 4},
    "epochs": 20,
    "batch_size": 16,
    "lr": 0.0001,
    "weight_decay": 0.05,
    "train_scope": "all",
    "classification_weight": 1.0,
    "contrastive_weight": 0.0,
    "two_views": False,
    "contrastive_mode": "supervised",
    "temperature": 0.1,
    "memory_bank_size": 0,
    "precision": "fp32",
    "seed": 42,
    "output": "runs/dinov3",
    "resume": None,
    "init_checkpoint": None,
    "wandb": None,
    "grad_clip": 1.0,
    "optimization": OPTIMIZATION_DEFAULTS,
}


def resolve_config(path):
    supplied = read_config(path)
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = {**DEFAULTS, **supplied}
    for key in ("model", "data", "optimization"):
        config[key] = {**DEFAULTS[key], **supplied.get(key, {})}
    return config


def validate_config(config):
    validate_optimization(config)
    if not isinstance(config["two_views"], bool):
        raise ValueError("two_views must be a boolean")
    if config["train_scope"] not in ("all", "backbone", "head"):
        raise ValueError("Invalid train_scope")
    if config["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    if config["contrastive_mode"] not in ("supervised", "instance"):
        raise ValueError("contrastive_mode must be supervised or instance")
    if config["classification_weight"] < 0 or config["contrastive_weight"] < 0:
        raise ValueError("Loss weights cannot be negative")
    if config["classification_weight"] + config["contrastive_weight"] <= 0:
        raise ValueError("Enable at least one loss")
    if config["train_scope"] == "head" and (config["contrastive_weight"] or not config["classification_weight"]):
        raise ValueError("Head-only training requires classification loss and contrastive_weight=0")
    if (
        config["train_scope"] == "backbone"
        and config["classification_weight"]
        and not (config["init_checkpoint"] or config["resume"])
    ):
        raise ValueError("A frozen classifier requires init_checkpoint; otherwise set classification_weight=0")
    if min(config["batch_size"], config["epochs"], config["data"]["image_size"]) <= 0:
        raise ValueError("batch_size, epochs and image_size must be positive")
    if config["temperature"] <= 0 or config["lr"] <= 0 or config["grad_clip"] < 0:
        raise ValueError("temperature/lr must be positive and grad_clip nonnegative")


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(model, loader, accelerator):
    model.eval()
    totals = torch.zeros(3, device=accelerator.device, dtype=torch.float64)
    dtype = next(model.parameters()).dtype
    # ZeRO-2 keeps full weights on each rank, so uneven validation shards need no padding.
    for images, _, labels, _ in loader:
        images = images.to(accelerator.device, dtype=dtype, non_blocking=True)
        labels = labels.to(accelerator.device)
        with accelerator.autocast():
            output = model(images)
        totals[0] += functional.cross_entropy(output["logits"].float(), labels, reduction="sum")
        totals[1] += output["logits"].argmax(1).eq(labels).sum()
        totals[2] += len(labels)
    totals = accelerator.reduce(totals, reduction="sum")
    if totals[2] == 0:
        return {}
    return {"val/loss": (totals[0] / totals[2]).item(), "val/accuracy": (totals[1] / totals[2]).item()}


def train(config):
    config = {**DEFAULTS, **config, "optimization": {**OPTIMIZATION_DEFAULTS, **config.get("optimization", {})}}
    validate_config(config)
    accelerator = create_accelerator(config, config["grad_clip"])
    try:
        _train(config, accelerator)
    finally:
        accelerator.end_training()


def _train(config, accelerator):
    device = accelerator.device
    rank, world_size = accelerator.process_index, accelerator.num_processes
    random.seed(config["seed"] + rank)
    torch.manual_seed(config["seed"] + rank)
    data_config = config["data"]
    transform_args = {
        "size": data_config["image_size"],
        "mean": data_config.get("mean"),
        "std": data_config.get("std"),
        "augment": data_config.get("augment"),
        "custom_transform": data_config.get("custom_transform"),
    }
    contrastive = config["contrastive_weight"] > 0
    two_views = config["two_views"] or contrastive
    dataset = FolderViews(data_config["train"], image_transform(**transform_args), two_views)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config["seed"], drop_last=True
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        drop_last=True,
        num_workers=data_config["workers"],
        pin_memory=device.type == "cuda",
    )
    if not len(loader):
        raise ValueError("Training set must contain at least batch_size * world_size images")
    validation = None
    if data_config.get("val"):
        val_args = {**transform_args, "augment": None}
        val_dataset = FolderViews(data_config["val"], image_transform(**val_args))
        if val_dataset.classes != dataset.classes:
            raise ValueError("Validation folders must have the same class names as training folders")
        validation = DataLoader(
            Subset(val_dataset, list(range(rank, len(val_dataset), world_size))),
            batch_size=config["batch_size"],
            num_workers=data_config["workers"],
        )
    resume = load_weights(config["resume"]) if config["resume"] else None
    if resume:
        if resume["format_version"] != CHECKPOINT_VERSION:
            raise ValueError("Resume requires a Transformers-backed checkpoint")
        resume_config = {
            **DEFAULTS,
            **resume["config"],
            "optimization": {**OPTIMIZATION_DEFAULTS, **resume["config"].get("optimization", {})},
        }
        mutable = {"resume", "output", "wandb", "epochs", "init_checkpoint"}
        if any(config[key] != resume_config[key] for key in config if key not in mutable):
            raise ValueError("Resume requires the original training configuration (except epochs/output/wandb)")
        if resume["classes"] != dataset.classes or resume["world_size"] != world_size:
            raise ValueError("Resume requires unchanged class mapping and world size")
    num_classes = len(dataset.classes) if config["classification_weight"] else 0
    initial = load_weights(config["init_checkpoint"]) if config["init_checkpoint"] and not resume else None
    model_config = resume["model_config"] if resume else config["model"]
    if initial:
        model_config = {**model_config, "backbone_config": initial["model_config"]["backbone_config"]}
    model = FineTuneModel(model_config, num_classes, initialize=not bool(resume or initial))
    if resume:
        model.load_state_dict(resume["model"], strict=True)
    elif initial:
        model.load_state_dict(initial["model"], strict=True)
    model.set_train_scope(config["train_scope"], contrastive)
    model.to(device)
    ema_enabled = config["optimization"]["ema_decay"] is not None
    ema = ModelEMA(model.state_dict()) if ema_enabled and rank == 0 else None
    bank = MemoryBank(config["memory_bank_size"], config["model"]["projection_dim"]).to(device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    schedule = TrainingSchedule(config, len(loader))
    runner, optimizer = accelerator.prepare(model, optimizer)
    start_epoch, global_step = 0, 0
    if resume:
        bank.load_state_dict(resume["bank"])
        start_epoch, global_step = resume["epoch"] + 1, resume["global_step"]
        schedule.load_state_dict(resume["schedule"])
        if ema is not None:
            ema.load_state_dict(resume["ema"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    run_id = resume["wandb_id"] if resume else uuid.uuid4().hex
    if config["wandb"] and rank == 0:
        import wandb

        options = dict(config["wandb"])
        options.pop("upload_checkpoint", None)
        wandb_run = wandb.init(**options, config=config, id=run_id, resume="allow")
    if rank == 0:
        (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        (output / "classes.json").write_text(
            json.dumps(dataset.classes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "parameter_groups.json").write_text(
            json.dumps(
                [{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups],
                indent=2,
            ),
            encoding="utf-8",
        )
    if resume:
        accelerator.load_state(Path(config["resume"]).resolve().parent / resume["accelerate_state"])
    try:
        for epoch in range(start_epoch, config["epochs"]):
            sampler.set_epoch(epoch)
            runner.train()
            totals = torch.zeros(5, device=device, dtype=torch.float64)
            for first, second, labels, ids in loader:
                base_lr, weight_decay = schedule.apply(optimizer)
                ema_decay = schedule.ema_decay() if ema_enabled else None
                images = torch.cat((first, second)) if two_views else first
                images = images.to(device, dtype=next(model.parameters()).dtype, non_blocking=True)
                labels, ids = labels.to(device), ids.to(device)
                targets = labels.repeat(2) if two_views else labels
                sample_ids = ids.repeat(2) if two_views else ids
                optimizer.zero_grad(set_to_none=True)
                with accelerator.autocast():
                    result = runner(images, contrastive=contrastive)
                    classification = (
                        functional.cross_entropy(
                            result["logits"].float(), targets, label_smoothing=config["optimization"]["label_smoothing"]
                        )
                        if num_classes
                        else images.new_zeros(())
                    )
                    metric_loss = (
                        contrastive_loss(
                            result["projections"],
                            targets,
                            sample_ids,
                            bank,
                            config["temperature"],
                            config["contrastive_mode"] == "supervised",
                        )
                        if contrastive
                        else images.new_zeros(())
                    )
                    loss = classification * config["classification_weight"] + metric_loss * config["contrastive_weight"]
                finite = torch.isfinite(loss).to(torch.int)
                if distributed():
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("Nonfinite loss on at least one rank")
                # Under ZeRO-2, backward also runs the engine's clipping and optimizer step.
                accelerator.backward(loss)
                if config["grad_clip"]:
                    accelerator.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
                updated = not accelerator.optimizer_step_was_skipped
                if updated and ema_enabled:
                    if ema is not None:
                        ema.update(model.state_dict(), ema_decay)
                if contrastive and updated:
                    bank.enqueue(result["projections"], targets, sample_ids)
                if updated:
                    schedule.step += 1
                count = len(targets)
                totals[0] += loss.detach() * count
                totals[1] += classification.detach() * count
                totals[2] += metric_loss.detach() * count
                if num_classes:
                    totals[3] += result["logits"].argmax(1).eq(targets).sum()
                totals[4] += count
                global_step += 1
                if rank == 0 and (global_step == 1 or global_step % 50 == 0):
                    progress = {
                        "unix_time": time.time(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "optimizer_steps": schedule.step,
                        "loss": loss.detach().item(),
                        "classification_loss": classification.detach().item(),
                        "contrastive_loss": metric_loss.detach().item(),
                        "lr": base_lr,
                        "original_images_per_global_batch": len(first) * world_size,
                        "views_per_original": 2 if two_views else 1,
                        "views_per_global_batch": len(images) * world_size,
                    }
                    with (output / "steps.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(progress) + "\n")
                    print(json.dumps({"stage": "train_step", **progress}), flush=True)
            totals = accelerator.reduce(totals, reduction="sum")
            metrics = {
                "epoch": epoch + 1,
                "train/loss": (totals[0] / totals[4]).item(),
                "train/classification_loss": (totals[1] / totals[4]).item(),
                "train/contrastive_loss": (totals[2] / totals[4]).item(),
                "train/lr": base_lr,
                "train/lr_min": min(group["lr"] for group in optimizer.param_groups),
                "train/lr_max": max(group["lr"] for group in optimizer.param_groups),
                "train/weight_decay": weight_decay,
                "train/optimizer_steps": schedule.step,
                "train/memory_bank_size": int(bank.size),
            }
            if ema is not None:
                metrics["train/ema_updates"] = ema.updates
                metrics["train/ema_decay"] = ema_decay
            if num_classes:
                metrics["train/accuracy"] = (totals[3] / totals[4]).item()
            if validation is not None and num_classes:
                metrics.update(evaluate(model, validation, accelerator))
            # All ranks save optimizer shards and RNG; last.pt remains a standalone model export.
            accelerator.save_state(output / "state")
            weights = accelerator.get_state_dict(runner)
            if rank == 0:
                payload = {
                    "format_version": CHECKPOINT_VERSION,
                    "model": weights,
                    "model_config": model.model_config,
                    "num_classes": num_classes,
                    "classes": dataset.classes,
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                    "world_size": world_size,
                    "bank": bank.state_dict(),
                    "schedule": schedule.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None,
                    "wandb_id": run_id,
                    "accelerate_state": "state",
                }
                atomic_save(payload, output / "last.pt")
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
                if wandb_run:
                    wandb_run.log(metrics, step=global_step)
            accelerator.wait_for_everyone()
        if wandb_run and config["wandb"].get("upload_checkpoint", False) and (output / "last.pt").exists():
            import wandb

            artifact = wandb.Artifact(f"dinov3-{run_id}", type="model")
            artifact.add_file(str(output / "last.pt"))
            artifact.add_file(str(output / "classes.json"))
            wandb_run.log_artifact(artifact)
    finally:
        if wandb_run:
            wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description="DINOv3 ImageFolder training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = resolve_config(args.config)
    if args.resume:
        config["resume"] = args.resume
    train(config)


if __name__ == "__main__":
    main()
