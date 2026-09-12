import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import uuid

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .contrastive import MemoryBank, contrastive_loss, distributed
from .data import FolderViews, image_transform
from .model import FineTuneModel, load_weights, read_config, state_dict_from
from .optimization import (
    OPTIMIZATION_DEFAULTS,
    ModelEMA,
    TrainingSchedule,
    parameter_groups,
    validate_optimization,
)


DEFAULTS = {
    "model": {"name": "dinov3_vits16", "projection_dim": 128},
    "data": {"train": "base", "val": None, "image_size": 224, "workers": 4},
    "epochs": 20,
    "batch_size": 16,
    "lr": 0.0001,
    "weight_decay": 0.05,
    "train_scope": "all",
    "classification_weight": 1.0,
    "contrastive_weight": 0.0,
    "contrastive_mode": "supervised",
    "temperature": 0.1,
    "memory_bank_size": 0,
    "precision": "fp32",
    "seed": 42,
    "output": "runs/dinov3",
    "resume": None,
    "init_checkpoint": None,
    "deepspeed": None,
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


def rng_state():
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def autocast_context(device, precision):
    return torch.autocast(
        device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16, enabled=precision != "fp32"
    )


def reduce_totals(totals):
    if distributed():
        dist.all_reduce(totals)
    return totals


@torch.no_grad()
def evaluate(model, loader, device, precision, synchronized=False, image_size=224):
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    iterations = torch.tensor(len(loader), device=device)
    if synchronized and distributed():
        dist.all_reduce(iterations, op=dist.ReduceOp.MAX)
    iterator = iter(loader)
    for _ in range(int(iterations)):
        batch = next(iterator, None)
        if batch is None:
            images = torch.zeros(1, 3, image_size, image_size, device=device)
            labels = None
        else:
            images, _, labels, _ = batch
            images, labels = images.to(device), labels.to(device)
        if synchronized:
            images = images.to(dtype=next(model.parameters()).dtype)
        with autocast_context(device, precision):
            output = model(images)
        if labels is not None and "logits" in output:
            totals[0] += functional.cross_entropy(output["logits"].float(), labels, reduction="sum")
            totals[1] += output["logits"].argmax(1).eq(labels).sum()
            totals[2] += len(labels)
    totals = reduce_totals(totals)
    if totals[2] == 0:
        return {}
    return {"val/loss": (totals[0] / totals[2]).item(), "val/accuracy": (totals[1] / totals[2]).item()}


def train(config):
    config = {**DEFAULTS, **config, "optimization": {**OPTIMIZATION_DEFAULTS, **config.get("optimization", {})}}
    validate_config(config)
    use_deepspeed = bool(config["deepspeed"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if device.type == "cpu" and config["precision"] == "fp16":
        raise ValueError("CPU training supports fp32 or bf16")
    if use_deepspeed:
        if device.type != "cuda":
            raise ValueError("This DeepSpeed entry point requires CUDA on Linux/WSL2")
        import deepspeed

        deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)
    elif world_size > 1 and not distributed():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if distributed() else 0
    world_size = dist.get_world_size() if distributed() else 1
    if use_deepspeed and (not distributed() or world_size < 1):
        raise RuntimeError("DeepSpeed did not initialize a distributed process group")
    if use_deepspeed and device.type == "cuda" and local_rank >= torch.cuda.device_count():
        raise ValueError(f"LOCAL_RANK={local_rank} exceeds visible CUDA devices ({torch.cuda.device_count()})")
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
    dataset = FolderViews(data_config["train"], image_transform(**transform_args), contrastive)
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
    legacy_resume = bool(resume and "schedule" not in resume)
    legacy_groups = legacy_resume or bool(resume and resume.get("legacy_parameter_groups", False))
    if resume:
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
    model = FineTuneModel(config["model"], num_classes, initialize=not bool(resume))
    if resume:
        model.load_state_dict(resume["model"], strict=True)
    elif config["init_checkpoint"]:
        model.load_state_dict(state_dict_from(load_weights(config["init_checkpoint"])), strict=True)
    model.set_train_scope(config["train_scope"], contrastive)
    model.to(device)
    ema_enabled = config["optimization"]["ema_decay"] is not None
    ema = ModelEMA(model.state_dict()) if ema_enabled and rank == 0 else None
    bank = MemoryBank(config["memory_bank_size"], config["model"]["projection_dim"]).to(device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config, legacy=legacy_groups),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    schedule = TrainingSchedule(config, len(loader))
    scaler = torch.amp.GradScaler(device.type, enabled=config["precision"] == "fp16" and not use_deepspeed)
    engine = None
    if use_deepspeed:
        ds_config = (
            read_config(config["deepspeed"]) if isinstance(config["deepspeed"], str) else dict(config["deepspeed"])
        )
        expected = {
            "train_micro_batch_size_per_gpu": config["batch_size"],
            "gradient_accumulation_steps": 1,
            "train_batch_size": config["batch_size"] * world_size,
        }
        for key, value in expected.items():
            if key in ds_config and ds_config[key] != value:
                raise ValueError(f"DeepSpeed {key} must equal {value}")
            ds_config[key] = value
        if "optimizer" in ds_config or "scheduler" in ds_config:
            raise ValueError(
                "Set optimizer in the training config; DeepSpeed optimizer/scheduler overrides are unsupported"
            )
        for precision in ("fp16", "bf16"):
            enabled = config["precision"] == precision
            if precision in ds_config and ds_config[precision].get("enabled", False) != enabled:
                raise ValueError("DeepSpeed precision must match training precision")
            ds_config[precision] = {**ds_config.get(precision, {}), "enabled": enabled}
        ds_config["gradient_clipping"] = config["grad_clip"]
        engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)
        runner = engine
    elif distributed():
        runner = DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True
        )
    else:
        runner = model
    start_epoch, global_step = 0, 0
    if resume:
        bank.load_state_dict(resume["bank"])
        if engine:
            checkpoint_dir = Path(config["resume"]).resolve().parent / "deepspeed"
            loaded, _ = engine.load_checkpoint(str(checkpoint_dir), tag=resume["deepspeed_tag"])
            if loaded is None:
                raise ValueError("DeepSpeed optimizer checkpoint is missing")
        else:
            optimizer.load_state_dict(resume["optimizer"])
            scaler.load_state_dict(resume["scaler"])
        start_epoch, global_step = resume["epoch"] + 1, resume["global_step"]
        if legacy_resume:
            schedule.step = global_step
        else:
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
        restore_rng(resume["rng"][rank])
    try:
        for epoch in range(start_epoch, config["epochs"]):
            sampler.set_epoch(epoch)
            runner.train()
            totals = torch.zeros(5, device=device, dtype=torch.float64)
            for first, second, labels, ids in loader:
                base_lr, weight_decay = schedule.apply(optimizer)
                ema_decay = schedule.ema_decay() if ema_enabled else None
                images = torch.cat((first, second)) if contrastive else first
                images = images.to(device, non_blocking=True)
                labels, ids = labels.to(device), ids.to(device)
                targets = labels.repeat(2) if contrastive else labels
                sample_ids = ids.repeat(2) if contrastive else ids
                if engine:
                    images = images.to(dtype=next(model.parameters()).dtype)
                with autocast_context(device, config["precision"]):
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
                if engine:
                    skipped_before = engine.skipped_steps
                    engine.backward(loss)
                    engine.step()
                    updated = engine.skipped_steps == skipped_before
                else:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if config["grad_clip"]:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    updated = scaler.get_scale() >= scale_before
                if updated and ema_enabled:
                    context = (
                        deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=None)
                        if engine and engine.zero_optimization_stage() == 3
                        else nullcontext()
                    )
                    with context:
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
            totals = reduce_totals(totals)
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
                metrics.update(
                    evaluate(
                        engine if engine else model,
                        validation,
                        device,
                        config["precision"],
                        synchronized=bool(engine),
                        image_size=data_config["image_size"],
                    )
                )
            states = [None] * world_size
            if distributed():
                dist.all_gather_object(states, rng_state())
            else:
                states[0] = rng_state()
            tag = f"epoch_{epoch + 1:04d}"
            if engine:
                engine.save_checkpoint(str(output / "deepspeed"), tag=tag)
            weights = (
                engine._zero3_consolidated_16bit_state_dict()
                if engine and engine.zero_optimization_stage() == 3
                else model.state_dict()
            )
            if rank == 0:
                payload = {
                    "format_version": 2,
                    "legacy_parameter_groups": legacy_groups,
                    "model": weights,
                    "model_config": config["model"],
                    "num_classes": num_classes,
                    "classes": dataset.classes,
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                    "world_size": world_size,
                    "bank": bank.state_dict(),
                    "optimizer": None if engine else optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "schedule": schedule.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None,
                    "rng": states,
                    "wandb_id": run_id,
                    "deepspeed_tag": tag if engine else None,
                }
                atomic_save(payload, output / "last.pt")
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
                if wandb_run:
                    wandb_run.log(metrics, step=global_step)
            if distributed():
                dist.barrier()
            restore_rng(states[rank])
        if wandb_run and config["wandb"].get("upload_checkpoint", False) and (output / "last.pt").exists():
            import wandb

            artifact = wandb.Artifact(f"dinov3-{run_id}", type="model")
            artifact.add_file(str(output / "last.pt"))
            artifact.add_file(str(output / "classes.json"))
            wandb_run.log_artifact(artifact)
    finally:
        if wandb_run:
            wandb_run.finish()
        if distributed():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="DINOv3 ImageFolder training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    config = resolve_config(args.config)
    if args.resume:
        config["resume"] = args.resume
    train(config)


if __name__ == "__main__":
    main()
