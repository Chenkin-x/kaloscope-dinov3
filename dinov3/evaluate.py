"""Fixed train-gallery/dev-query retrieval; never select checkpoints on test data."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Subset


def content_mask(qids, gids, content):
    tokens, rows, columns, known = {}, [], [], []
    for i, index in enumerate(np.r_[qids, gids]):
        characters, copyrights = content[str(int(index))]
        known.append(bool(characters.strip() and copyrights.strip()))
        tags = {"c:" + x for x in characters.split()} | {"p:" + x for x in copyrights.split()}
        for tag in tags:
            tokens.setdefault(tag, len(tokens))
            rows.append(i)
            columns.append(tokens[tag])
    matrix = csr_matrix((np.ones(len(rows), dtype="int32"), (rows, columns)), shape=(len(known), len(tokens)))
    n = len(qids)
    overlap = (matrix[:n] @ matrix[n:].T).toarray() > 0
    known = np.array(known)
    return (~overlap) & known[:n, None] & known[None, n:]


def load_panel(root, metadata, world=1):
    root = Path(root)
    panel = {role: np.load(root / f"{role}.npy", allow_pickle=False) for role in ("gallery", "query")}
    for role, split in (("gallery", 0), ("query", 1)):
        ids = panel[role]
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < world:
            raise ValueError(f"{role} must contain at least world_size integer indices")
        if (ids < 0).any() or (ids >= len(metadata["split"])).any() or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Invalid or repeated {role} indices")
        if not (metadata["split"][ids] == split).all() or (metadata["group_ids"][ids] < 0).any():
            raise ValueError("Evaluation requires assigned train-gallery/dev-query images; test is sealed")
    if len(panel["gallery"]) < 10:
        raise ValueError("Hit@10 evaluation needs at least 10 gallery images")
    if np.intersect1d(metadata["work_family_id"][panel["gallery"]], metadata["work_family_id"][panel["query"]]).size:
        raise ValueError("Gallery and query contain the same work family")
    panel["content"] = json.loads((root / "content.json").read_text())
    for index in np.r_[panel["gallery"], panel["query"]]:
        value = panel["content"].get(str(int(index)))
        if not isinstance(value, list) or len(value) != 2 or not all(isinstance(x, str) for x in value):
            raise ValueError("content.json must map each panel index to [character tags, copyright tags]")
    return panel


@torch.no_grad()
def extract(model, dataset, indices, batch, workers, precision):
    distributed = dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    device = next(model.parameters()).device
    loader = DataLoader(
        Subset(dataset, [int(i) for i in indices[rank::world]]),
        batch_size=batch,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    values = []
    for images, _ in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            values.append(model(images.to(device, non_blocking=True)).float().cpu().numpy())
    local = np.concatenate(values)
    if not distributed:
        return local
    results = [None] * world if rank == 0 else None
    dist.gather_object(local, results, dst=0)
    return np.concatenate(results) if rank == 0 else None


@torch.no_grad()
def evaluate(model, dataset, metadata, panel, config):
    distributed = dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    values, ids = {}, {}
    try:
        for role in ("gallery", "query"):
            ii = panel[role]
            values[role] = extract(
                model, dataset, ii, config["eval_batch_size"], config["eval_workers"], config["precision"]
            )
            ids[role] = np.concatenate([ii[r::world] for r in range(world)])
        result = None
        if rank == 0:
            artist, days, groups = (metadata[k] for k in ("artist_ids", "days", "group_ids"))
            gids, qids = ids["gallery"], ids["query"]
            allowed = content_mask(qids, gids, panel["content"])
            same = artist[qids, None] == artist[gids][None, :]
            positive = same & (np.abs(days[qids, None] - days[gids][None, :]) <= config["window_days"])
            result = {}
            for name, selection in (
                ("backbone", slice(0, model.feature_dim)),
                ("projector", slice(model.feature_dim, None)),
            ):
                gallery = torch.as_tensor(values["gallery"][:, selection], device=device)
                queries = torch.as_tensor(values["query"][:, selection], device=device)
                sim = F.normalize(queries, dim=1) @ F.normalize(gallery, dim=1).T
                metrics = {}
                if name == "projector":
                    centered = gallery - gallery.mean(0)
                    singular = torch.linalg.svdvals(centered)
                    probability = singular.square() / singular.square().sum().clamp_min(1e-12)
                    metrics["effective_rank"] = float((-(probability * probability.clamp_min(1e-12).log()).sum()).exp())
                    metrics["feature_std"] = float(centered.std(0).mean())
                for condition, mask in (("all", np.ones_like(positive)), ("cross_content", allowed)):
                    pos = torch.as_tensor(positive & mask, device=device)
                    eligible = pos.any(1) & torch.as_tensor(mask.sum(1) >= 10, device=device)
                    nn = sim.masked_fill(~torch.as_tensor(mask, device=device), -torch.inf).topk(10, dim=1).indices
                    hits = pos.gather(1, nn)
                    artist_hits = torch.as_tensor(same, device=device).gather(1, nn)
                    group_hits = torch.as_tensor(groups[qids, None] == groups[gids][None, :], device=device).gather(
                        1, nn
                    )
                    qa = artist[qids][eligible.cpu().numpy()]
                    hit10 = hits[eligible].any(1).float().cpu().numpy()
                    unique = np.unique(qa)
                    if not len(unique):
                        raise ValueError("No eligible validation artists for " + condition)
                    metrics[condition] = {
                        "queries": len(qa),
                        "artists": len(unique),
                        "temporal_hit_at_1": float(hits[eligible, 0].float().mean()),
                        "temporal_hit_at_10": float(hit10.mean()),
                        "macro_temporal_hit_at_10": float(np.mean([hit10[qa == aid].mean() for aid in unique])),
                        "temporal_precision_at_10": float(hits[eligible].float().mean()),
                        "artist_hit_at_1": float(artist_hits[eligible, 0].float().mean()),
                        "year_group_hit_at_10": float(group_hits[eligible].any(1).float().mean()),
                    }
                result[name] = metrics
            result["selection_score"] = result["projector"]["cross_content"]["macro_temporal_hit_at_10"]
            result["healthy"] = bool(
                np.isfinite(result["selection_score"])
                and result["projector"]["effective_rank"] >= config["min_effective_rank"]
                and result["projector"]["feature_std"] > config["min_feature_std"]
            )
            result["test_used"] = False
        if distributed:
            packed = [result]
            dist.broadcast_object_list(packed, src=0)
            result = packed[0]
        return result
    finally:
        model.train(was_training)
