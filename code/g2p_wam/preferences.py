"""Outcome-stratified confidence filtering of caller-defined episode groups."""
from collections import defaultdict
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class EpisodeScore:
    episode_id: str
    group_id: str
    success: bool
    confidence: float


@dataclass(frozen=True)
class PreferencePair:
    group_id: str
    winner_id: str
    loser_id: str


def build_pairs(records, *, pairs_per_group, keep_fraction=0.5, seed=0):
    """Rank-filter each outcome pool, then sample with replacement.

    Group IDs are provided by the caller, not evidence of identical resets.
    Outcome determines order; no minimum cross-pool confidence gap is enforced.
    Groups missing either outcome are skipped. Empty input returns an empty list.
    """
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must lie in (0,1]")
    if not isinstance(pairs_per_group, int) or pairs_per_group < 0:
        raise ValueError("pairs_per_group must be a nonnegative integer")
    groups, seen = defaultdict(list), set()
    for record in records:
        if not record.episode_id or not record.group_id:
            raise ValueError("Episode and group IDs must be nonempty")
        if record.episode_id in seen:
            raise ValueError("Episode IDs must be unique in the supplied pool")
        if record.success not in (False, True) or not math.isfinite(record.confidence):
            raise ValueError("Expected binary outcomes and finite confidence")
        seen.add(record.episode_id)
        groups[record.group_id].append(record)
    generator, pairs = random.Random(seed), []
    for group_id in sorted(groups):
        winners = sorted((r for r in groups[group_id] if r.success),
                         key=lambda r: (-r.confidence, r.episode_id))
        losers = sorted((r for r in groups[group_id] if not r.success),
                        key=lambda r: (r.confidence, r.episode_id))
        if not winners or not losers:
            continue
        winners = winners[:max(1, math.floor(len(winners) * keep_fraction))]
        losers = losers[:max(1, math.floor(len(losers) * keep_fraction))]
        for _ in range(pairs_per_group):
            pairs.append(PreferencePair(group_id, generator.choice(winners).episode_id,
                                         generator.choice(losers).episode_id))
    return pairs
