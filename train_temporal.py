"""Accelerate artist-temporal training; default launcher uses DeepSpeed ZeRO-2."""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.utils.data import DataLoader

from dinov3.config import resolve_config, validate_config
from dinov3.control import Convergence
from dinov3.data import Originals, load_metadata
from dinov3.evaluate import evaluate, load_panel
from dinov3.loss import masks, sigmoid_loss
from dinov3.model import CHECKPOINT_VERSION, Encoder
from dinov3.runtime import create_accelerator
from dinov3.sampling import PairPool


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def fingerprint(config):
    """Hash every consumed manifest, not just a possibly stale summary report."""
    paths = [Path(config["paths"])]
    paths += [
        Path(config["metadata"]) / f"{name}.npy"
        for name in ("artist_ids", "days", "work_family_id", "group_ids", "split")
    ]
    paths += [Path(config["panel"]) / name for name in ("gallery.npy", "query.npy", "content.json")]
    result = {}
    for path in paths:
        with path.open("rb") as stream:
            result[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def rng_state(device):
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def train(config, resume=False, max_steps=0):
    validate_config(config)
    accelerator = create_accelerator(config, config["gradient_clip"])
    try:
        _train(config, resume, max_steps, accelerator)
    finally:
        accelerator.end_training()


def _train(config, resume, max_steps, accelerator):
    device = accelerator.device
    rank, world = accelerator.process_index, accelerator.num_processes
    distributed = world > 1
    global_batch = config["batch_size"] * world
    if global_batch < 4 or global_batch % 4:
        raise ValueError("batch_size * world_size must be positive and divisible by 4")
    dataset = Originals(config["paths"], config["image_size"])
    metadata = load_metadata(config["metadata"], len(dataset))
    panel = load_panel(config["panel"], metadata, world)
    n = int(((metadata["split"] == 0) & (metadata["group_ids"] >= 0)).sum())
    if not n:
        raise ValueError("No assigned training images")
    steps_per_coverage = math.ceil(n / (global_batch // 4))
    signature = {"config": config, "world_size": world, "manifests": fingerprint(config)}
    output = Path(config["output"])
    checkpoint = None
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("signature") != signature:
            raise ValueError("Resume requires unchanged configuration, world size and data manifests")
        if checkpoint.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError("Resume requires a Transformers-backed checkpoint")
    if rank == 0:
        if not resume:
            output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "config.json", signature)
        atomic_json(output / "status.json", {"status": "initializing", "time": time.time()})

    accelerator.wait_for_everyone()
    torch.manual_seed(config["seed"])
    model_config = checkpoint["model_config"] if checkpoint else config["model"]
    model = Encoder(model_config, initialize=not resume).to(device)
    model.bias.data.fill_(-math.log(global_batch - 2))
    model.set_train_from_block(config["train_from_block"])
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    groups = []
    for parameters, key in (
        ([p for p in model.backbone.parameters() if p.requires_grad], "backbone_lr"),
        (list(model.projector.parameters()), "projector_lr"),
        ([model.log_temperature, model.bias], "temperature_bias_lr"),
    ):
        groups.append(
            {
                "params": parameters,
                "lr": config[key],
                "base_lr": config[key],
                "weight_decay": 0.0 if key == "temperature_bias_lr" else config["weight_decay"],
            }
        )
    optimizer = torch.optim.AdamW(groups)
    runner, optimizer = accelerator.prepare(model, optimizer)
    controller = Convergence(**config["early_stop"])
    step = epoch = offset = 0
    torch.manual_seed(config["seed"] + rank)
    random.seed(config["seed"] + rank)
    np.random.seed((config["seed"] + rank) % 2**32)
    if checkpoint:
        controller.__dict__.update(checkpoint["controller"])
        step, epoch, offset = (checkpoint[k] for k in ("step", "epoch", "offset"))
        accelerator.load_state(output / checkpoint["accelerate_state"])
        del checkpoint
    cache = output / "orders"
    if rank == 0:
        cache.mkdir(exist_ok=True)
    accelerator.wait_for_everyone()
    pool = None
    start, start_step = time.time(), step

    def save(best=False):
        accelerator.save_state(output / "state")
        weights = accelerator.get_state_dict(runner)
        if rank == 0:
            payload = {
                "format_version": CHECKPOINT_VERSION,
                "model": weights,
                "model_config": model.model_config,
                "signature": signature,
                "controller": vars(controller),
                "step": step,
                "epoch": epoch,
                "offset": offset,
                "accelerate_state": "state",
            }
            for name in ["last", "best"] if best else ["last"]:
                temporary = output / f"{name}.tmp"
                torch.save(payload, temporary)
                temporary.replace(output / f"{name}.pt")
        accelerator.wait_for_everyone()

    def validate():
        state = rng_state(device)
        try:
            metrics = evaluate(model, dataset, metadata, panel, config)
        finally:
            restore_rng(state, device)
        action = controller.update(metrics["selection_score"], step / steps_per_coverage, metrics["healthy"])
        if rank == 0:
            item = dict(step=step, coverage_passes=step / steps_per_coverage, metrics=metrics, **action)
            with (output / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps(item) + "\n")
            print("VALIDATION " + json.dumps(item), flush=True)
        save(action["improved"])
        return action["stop_reason"]

    if not resume:
        validate()
    reason = "max_steps" if max_steps and step >= max_steps else None
    while epoch < config["max_coverage_passes"] and not reason:
        path = cache / f"epoch_{epoch}.npy"
        if rank == 0 and not path.exists():
            atomic_json(output / "status.json", {"status": "preparing_epoch_order", "epoch": epoch, "step": step})
            if pool is None:
                pool = PairPool(config["metadata"], config["window_days"])
                atomic_json(output / "sampling_report.json", pool.report)
            temporary = path.with_suffix(".building.npy")
            order = np.lib.format.open_memmap(
                temporary, mode="w+", dtype="int64", shape=(steps_per_coverage, world, config["batch_size"])
            )
            for i, batch in enumerate(pool.batches(epoch, config["seed"], global_batch)):
                order[i] = batch.reshape(world, config["batch_size"])
            order.flush()
            del order
            temporary.replace(path)
        accelerator.wait_for_everyone()
        order = np.load(path, mmap_mode="r")
        if order.shape != (steps_per_coverage, world, config["batch_size"]):
            raise ValueError("Invalid cached epoch order")
        generator = torch.Generator().manual_seed(config["seed"] + epoch + rank)
        loader = DataLoader(
            dataset,
            sampler=order[offset:, rank, :].reshape(-1).tolist(),
            batch_size=config["batch_size"],
            num_workers=config["workers"],
            pin_memory=device.type == "cuda",
            persistent_workers=config["workers"] > 0,
            generator=generator,
        )
        runner.train()
        for images, indices in loader:
            step += 1
            offset += 1
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * min(1.0, step / max(1, config["warmup_steps"])) * controller.factor
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                features, temperature, bias = runner(images.to(device, non_blocking=True))
            features = torch.cat(all_gather(features.float())) if distributed else features.float()
            meta = torch.tensor(
                np.stack(
                    [
                        metadata["artist_ids"][indices.numpy()],
                        metadata["days"][indices.numpy()],
                        indices.numpy(),
                        metadata["work_family_id"][indices.numpy()],
                    ]
                ),
                device=device,
            )
            if distributed:
                parts = [torch.empty_like(meta) for _ in range(world)]
                dist.all_gather(parts, meta)
                meta = torch.cat(parts, 1)
            positive, negative = masks(*meta, window_days=config["window_days"])
            loss = sigmoid_loss(features, temperature, bias, positive, negative)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite sigmoid loss")
            accelerator.backward(loss)
            # DeepSpeed already clipped during backward; Accelerate returns that global norm.
            norm = accelerator.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            if not math.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            if rank == 0 and (step <= 5 or step % config["log_every"] == 0):
                item = {
                    "step": step,
                    "epoch": epoch,
                    "offset": offset,
                    "coverage_passes": step / steps_per_coverage,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(norm),
                    "temperature": float(temperature.detach()),
                    "bias": float(bias.detach()),
                    "positive_pairs": int(positive.sum()),
                    "negative_pairs": int(negative.sum()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "images_per_second": (step - start_step) * global_batch / (time.time() - start),
                    "time": time.time(),
                }
                with (output / "steps.jsonl").open("a") as stream:
                    stream.write(json.dumps(item) + "\n")
                atomic_json(output / "status.json", dict(status="running", **item))
                print(json.dumps(item), flush=True)
            if step % config["eval_every"] == 0:
                reason = validate()
            elif step % config["save_every"] == 0:
                save()
            requested = torch.tensor(int((output / "STOP").exists()), device=device)
            if distributed:
                dist.all_reduce(requested, op=dist.ReduceOp.MAX)
            if requested.item():
                reason = reason or "user_stop_file"
            if max_steps and step >= max_steps:
                reason = reason or "max_steps"
            if reason:
                break
        del loader, order
        if not reason:
            epoch += 1
            offset = 0
    reason = reason or "max_coverage_guard"
    save()
    if rank == 0:
        atomic_json(
            output / "status.json",
            {
                "status": "stopped",
                "termination": reason,
                "converged": reason == "validation_plateau",
                "step": step,
                "coverage_passes": step / steps_per_coverage,
                "time": time.time(),
                "test_used": False,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="Total step limit for a bounded run; 0 disables")
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be nonnegative")
    train(resolve_config(args.config), args.resume, args.max_steps)


if __name__ == "__main__":
    main()
