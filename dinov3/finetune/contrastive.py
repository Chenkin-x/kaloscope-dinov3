import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch import nn


def distributed():
    return dist.is_available() and dist.is_initialized()


def gather_detached(tensor):
    if not distributed():
        return tensor.detach()
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.detach().contiguous())
    return torch.cat(gathered)


class MemoryBank(nn.Module):
    def __init__(self, capacity, dimension):
        super().__init__()
        if capacity < 0:
            raise ValueError("memory_bank_size must be nonnegative")
        self.register_buffer("features", torch.zeros(capacity, dimension))
        self.register_buffer("labels", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer("ids", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer("size", torch.zeros((), dtype=torch.long))
        self.register_buffer("pointer", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, features, labels, ids):
        capacity = len(self.features)
        if not capacity:
            return
        features = gather_detached(functional.normalize(features.float(), dim=1))[-capacity:]
        labels = gather_detached(labels)[-capacity:]
        ids = gather_detached(ids)[-capacity:]
        indices = (torch.arange(len(features), device=features.device) + self.pointer) % capacity
        self.features[indices] = features
        self.labels[indices] = labels
        self.ids[indices] = ids
        self.pointer.copy_((self.pointer + len(features)) % capacity)
        self.size.copy_(torch.clamp(self.size + len(features), max=capacity))


def contrastive_loss(projections, labels, ids, bank, temperature=0.1, supervised=True):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    anchors = functional.normalize(projections.float(), dim=1)
    if distributed():
        from torch.distributed.nn.functional import all_gather

        candidates = torch.cat(all_gather(anchors.contiguous()))
    else:
        candidates = anchors
    rank = dist.get_rank() if distributed() else 0
    offset = rank * len(anchors)
    candidate_labels = gather_detached(labels)
    candidate_ids = gather_detached(ids)
    current_count = len(candidates)
    size = int(bank.size)
    candidates = torch.cat((candidates, bank.features[:size].detach().clone()))
    candidate_labels = torch.cat((candidate_labels, bank.labels[:size]))
    candidate_ids = torch.cat((candidate_ids, bank.ids[:size]))
    valid = torch.ones((len(anchors), len(candidates)), device=anchors.device, dtype=torch.bool)
    valid[
        torch.arange(len(anchors), device=anchors.device), offset + torch.arange(len(anchors), device=anchors.device)
    ] = False
    if supervised:
        positive = labels[:, None].eq(candidate_labels[None, :]) & valid
    else:
        positive = ids[:, None].eq(candidate_ids[None, :]) & valid
        valid[:, current_count:] &= ~ids[:, None].eq(candidate_ids[None, current_count:])
        positive &= valid
    with torch.autocast(anchors.device.type, enabled=False):
        logits = (anchors @ candidates.T) / temperature
    logits = logits.masked_fill(~valid, -torch.inf)
    counts = positive.sum(1)
    if not torch.all(counts > 0):
        raise ValueError("Every contrastive anchor needs a positive; supply two views per image")
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob.masked_fill(~positive, 0).sum(1) / counts).mean()
