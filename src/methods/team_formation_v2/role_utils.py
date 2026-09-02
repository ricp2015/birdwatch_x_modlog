"""Build role, ideology, familiarity, and alignment features."""

from __future__ import annotations

from collections import defaultdict
import heapq
from itertools import combinations

import numpy as np
import pandas as pd

ROLE_NAMES = ("periphery", "core")
FamiliarityKey = tuple[str, str, str]


def fit_ideology(train: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit the official one-factor Community Notes MF on the prefix only."""
    if train["username"].nunique() < 2 or train["item_id"].nunique() < 2:
        return {}

    # Delayed import keeps methods that only reuse topic helpers from paying the
    # Community Notes/Torch startup cost.
    from src.methods.community_notes import c, run_cn_mf

    _, rater_params, _ = run_cn_mf(train)
    users = rater_params[c.raterParticipantIdKey].astype(str)
    factors = pd.to_numeric(
        rater_params[c.internalRaterFactor1Key],
        errors="coerce",
    )
    return {
        user: np.asarray([float(factor)], dtype=float)
        for user, factor in zip(users, factors)
        if np.isfinite(factor)
    }


def fit_familiarity(
    train: pd.DataFrame,
) -> tuple[dict[FamiliarityKey, float], dict[str, float]]:
    """Count prior shared cases, Huddler's direct familiarity operationalization."""
    counts: defaultdict[FamiliarityKey, float] = defaultdict(float)
    for (community, _), group in train.groupby(["community", "item_id"], sort=False):
        users = sorted(group["username"].dropna().astype(str).unique())
        for left, right in combinations(users, 2):
            counts[(str(community), left, right)] += 1.0

    scales: dict[str, float] = defaultdict(lambda: 1.0)
    for (community, _, _), count in counts.items():
        scales[community] = max(scales[community], float(count))
    return dict(counts), dict(scales)


def _core_numbers(users: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    """Compute unweighted k-core numbers without an additional dependency."""
    neighbours = {user: set() for user in users}
    for left, right in edges:
        if left == right or left not in neighbours or right not in neighbours:
            continue
        neighbours[left].add(right)
        neighbours[right].add(left)

    degrees = {user: len(values) for user, values in neighbours.items()}
    heap = [(degree, user) for user, degree in degrees.items()]
    heapq.heapify(heap)
    removed: set[str] = set()
    core: dict[str, int] = {}
    while heap:
        degree, user = heapq.heappop(heap)
        if user in removed or degree != degrees[user]:
            continue
        removed.add(user)
        core[user] = degree
        for neighbour in neighbours[user]:
            if neighbour in removed or degrees[neighbour] <= degree:
                continue
            degrees[neighbour] -= 1
            heapq.heappush(heap, (degrees[neighbour], neighbour))
    return core


def assign_core_periphery_roles(
    profiles: pd.DataFrame,
    familiarity: dict[FamiliarityKey, float],
) -> pd.DataFrame:
    """Assign community-specific maximal-k-core versus periphery roles."""
    out = profiles.copy()
    out["core_number"] = 0
    out["core_score"] = 0.0
    out["core_position"] = "periphery"

    community_edges: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for community, left, right in familiarity:
        community_edges[community].append((left, right))

    for community, group in out.groupby("community", sort=False):
        users = group["username"].astype(str).tolist()
        numbers = _core_numbers(users, community_edges[str(community)])
        values = group["username"].astype(str).map(numbers).fillna(0).astype(int)
        maximum = int(values.max()) if len(values) else 0
        out.loc[group.index, "core_number"] = values.to_numpy()
        if maximum > 0:
            out.loc[group.index, "core_score"] = values.to_numpy(dtype=float) / maximum
            out.loc[group.index, "core_position"] = np.where(
                values.to_numpy() == maximum,
                "core",
                "periphery",
            )
    out["core_number"] = out["core_number"].astype(int)
    return out


def crowding_distance(
    front: list[int],
    objectives: list[tuple[float, ...]],
) -> dict[int, float]:
    """Return standard NSGA-II crowding distances for one front."""
    distance = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: float("inf") for index in front}
    for objective in range(len(objectives[0])):
        ordered = sorted(front, key=lambda index: objectives[index][objective])
        distance[ordered[0]] = distance[ordered[-1]] = float("inf")
        low = objectives[ordered[0]][objective]
        high = objectives[ordered[-1]][objective]
        if high == low:
            continue
        for position in range(1, len(ordered) - 1):
            before = objectives[ordered[position - 1]][objective]
            after = objectives[ordered[position + 1]][objective]
            distance[ordered[position]] += (after - before) / (high - low)
    return distance
