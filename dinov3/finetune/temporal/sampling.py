"""Complete uniform anchor traversal + artist/year-balanced anchors, each with a temporal mate."""

import itertools
from pathlib import Path

import numpy as np


class PairPool:
    def __init__(self, root, window_days=365):
        root = Path(root)
        self.artist = np.load(root / "artist_ids.npy", mmap_mode="r")
        self.day = np.load(root / "days.npy", mmap_mode="r").astype("int64", copy=False)
        self.family = np.load(root / "work_family_id.npy", mmap_mode="r")
        self.group = np.load(root / "group_ids.npy", mmap_mode="r")
        split = np.load(root / "split.npy", mmap_mode="r")
        self.indices = np.flatnonzero((split == 0) & (self.group >= 0))
        if not len(self.indices):
            raise ValueError("No assigned training images")
        self.sorted = self.indices[np.lexsort((self.day[self.indices], self.artist[self.indices]))]
        self.position = np.full(len(self.artist), -1, dtype="int64")
        self.position[self.sorted] = np.arange(len(self.sorted))
        a = self.artist[self.sorted]
        d = self.day[self.sorted]
        f = self.family[self.sorted]
        n = len(a)
        cuts = np.flatnonzero(np.r_[True, np.diff(a) != 0, True])
        self.lower = np.empty(n, dtype="int64")
        self.upper = np.empty(n, dtype="int64")
        for lo, hi in itertools.pairwise(cuts):
            self.lower[lo:hi] = lo + np.searchsorted(d[lo:hi], d[lo:hi] - window_days, side="left")
            self.upper[lo:hi] = lo + np.searchsorted(d[lo:hi], d[lo:hi] + window_days, side="right")
        # Closest distinct-family neighbours are just outside contiguous family runs.
        starts = np.flatnonzero(np.r_[True, (np.diff(a) != 0) | (f[1:] != f[:-1])])
        ends = np.r_[starts[1:], n]
        previous = np.repeat(starts - 1, ends - starts)
        following = np.repeat(ends, ends - starts)
        prev = np.clip(previous, 0, n - 1)
        nxt = np.clip(following, 0, n - 1)
        goodprev = (previous >= 0) & (a[prev] == a) & (f[prev] != f) & (np.abs(d - d[prev]) <= window_days)
        goodnext = (following < n) & (a[nxt] == a) & (f[nxt] != f) & (np.abs(d - d[nxt]) <= window_days)
        self.fallback = np.where(goodprev, prev, np.where(goodnext, nxt, -1))
        eligible = self.sorted[self.fallback >= 0]
        if not len(eligible):
            raise ValueError("No distinct-work temporal positive pairs in training data")
        order = eligible[np.argsort(self.group[eligible], kind="stable")]
        g = self.group[order]
        cuts = np.flatnonzero(np.r_[True, np.diff(g) != 0, True])
        self.group_pools = {}
        self.artist_groups = {}
        for lo, hi in itertools.pairwise(cuts):
            rows = order[lo:hi]
            gid = int(g[lo])
            aid = int(self.artist[rows[0]])
            self.group_pools[gid] = rows
            self.artist_groups.setdefault(aid, []).append(gid)
        self.balanced_artists = np.array(sorted(self.artist_groups))
        if not len(self.balanced_artists):
            raise ValueError("No distinct-work temporal positive pairs in training data")
        self.report = {
            "train_images": len(self.indices),
            "pair_eligible_images": len(eligible),
            "no_temporal_mate_images": len(self.indices) - len(eligible),
            "balanced_artists": len(self.balanced_artists),
            "balanced_groups": len(self.group_pools),
        }

    def balanced(self, rng):
        aid = int(self.balanced_artists[rng.integers(len(self.balanced_artists))])
        gs = self.artist_groups[aid]
        pool = self.group_pools[gs[rng.integers(len(gs))]]
        return int(pool[rng.integers(len(pool))])

    def mate(self, row, rng):
        p = self.position[row]
        fallback = self.fallback[p]
        if fallback < 0:
            # Keep the original in full coverage; use a different artist for the extra slot.
            for _ in range(100):
                other = int(self.indices[rng.integers(len(self.indices))])
                if self.artist[other] != self.artist[row]:
                    return other, False
            raise RuntimeError("No different-artist fallback")
        lo, hi = self.lower[p], self.upper[p]
        for _ in range(8):
            other = int(self.sorted[rng.integers(lo, hi)])
            if self.family[other] != self.family[row]:
                return other, True
        return int(self.sorted[fallback]), True

    def batches(self, epoch, seed, global_batch, limit=0):
        if global_batch < 4 or global_batch % 4:
            raise ValueError("global batch must be positive and divisible by 4")
        rng = np.random.default_rng(seed + epoch)
        uniform = rng.permutation(self.indices)
        take = global_batch // 4
        steps = (len(uniform) + take - 1) // take
        for step in range(min(steps, limit) if limit else steps):
            anchors = uniform[step * take : (step + 1) * take].tolist()
            if len(anchors) < take:
                anchors.extend(np.resize(uniform, take - len(anchors)).tolist())
            anchors.extend(self.balanced(rng) for _ in range(take))
            rows = []
            for anchor in anchors:
                rows.extend([anchor, self.mate(anchor, rng)[0]])
            # Pairs remain adjacent, but rank assignment is shuffled to avoid source bias.
            pairs = np.array(rows, dtype="int64").reshape(-1, 2)
            rng.shuffle(pairs, axis=0)
            yield pairs.reshape(-1)
