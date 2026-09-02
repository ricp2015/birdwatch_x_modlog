"""Optimize four team-composition objectives with NSGA-II."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from src.methods.team_formation_v2.role_utils import FamiliarityKey, crowding_distance


def _dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(a >= b for a, b in zip(left, right)) and any(
        a > b for a, b in zip(left, right)
    )


def nondominated_sort(objectives: list[tuple[float, ...]]) -> list[list[int]]:
    """Return ordinary Pareto fronts over all four specified objectives."""
    dominates = [set() for _ in objectives]
    dominated_count = [0] * len(objectives)
    fronts = [[]]
    for left_index, left in enumerate(objectives):
        for right_index, right in enumerate(objectives):
            if left_index == right_index:
                continue
            if _dominates(left, right):
                dominates[left_index].add(right_index)
            elif _dominates(right, left):
                dominated_count[left_index] += 1
        if dominated_count[left_index] == 0:
            fronts[0].append(left_index)
    rank = 0
    while fronts[rank]:
        following = []
        for left_index in fronts[rank]:
            for right_index in dominates[left_index]:
                dominated_count[right_index] -= 1
                if dominated_count[right_index] == 0:
                    following.append(right_index)
        rank += 1
        fronts.append(following)
    return fronts[:-1]


class CommunityNSGA2Optimizer:
    """Reuse community-static pairwise objectives for every incoming case."""

    def __init__(
        self,
        candidates: pd.DataFrame,
        ideology: dict[str, np.ndarray],
        familiarity: dict[FamiliarityKey, float],
        familiarity_scales: dict[str, float],
        team_size: int,
        population_size: int,
        generations: int,
    ):
        self.candidates = candidates.reset_index(drop=True)
        self.n_candidates = len(self.candidates)
        self.k = min(int(team_size), self.n_candidates)
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.community = str(self.candidates.iloc[0]["community"])
        users = self.candidates["username"].astype(str).tolist()

        ideology_dimension = len(next(iter(ideology.values()))) if ideology else 1
        vectors = np.vstack(
            [ideology.get(user, np.zeros(ideology_dimension)) for user in users]
        )
        known = np.asarray([user in ideology for user in users], dtype=bool)
        self.ideology_matrix = np.linalg.norm(
            vectors[:, None, :] - vectors[None, :, :],
            axis=2,
        )
        self.ideology_matrix *= known[:, None] & known[None, :]

        self.familiarity_matrix = np.zeros(
            (self.n_candidates, self.n_candidates),
            dtype=float,
        )
        scale = max(float(familiarity_scales.get(self.community, 1.0)), 1.0)
        for left in range(self.n_candidates):
            for right in range(left + 1, self.n_candidates):
                key = (self.community, *sorted((users[left], users[right])))
                value = familiarity.get(key, 0.0) / scale
                self.familiarity_matrix[left, right] = value
                self.familiarity_matrix[right, left] = value

        # Relative bands avoid imposing a universal agreement threshold on
        # communities with different moderator base rates.
        alignment = self.candidates["profile_alignment_local"].to_numpy(dtype=float)
        order = np.argsort(alignment, kind="stable")
        high_count = max(1, math.ceil(self.n_candidates / 3))
        mid_count = max(1, math.ceil(self.n_candidates / 3)) if self.n_candidates > 1 else 0
        self.high_alignment = np.zeros(self.n_candidates, dtype=bool)
        self.mid_alignment = np.zeros(self.n_candidates, dtype=bool)
        self.high_alignment[order[-high_count:]] = True
        mid_stop = self.n_candidates - high_count
        mid_start = max(0, mid_stop - mid_count)
        self.mid_alignment[order[mid_start:mid_stop]] = True

    def compose(
        self,
        expertise_values: np.ndarray,
        seed: int,
    ) -> tuple[tuple[int, ...], tuple[float, ...], int, float]:
        """Optimize all four objectives and return a closest-to-ideal compromise."""
        expertise = np.asarray(expertise_values, dtype=float)
        if len(expertise) != self.n_candidates:
            raise ValueError("Expertise length does not match the candidate pool")
        objective_cache: dict[tuple[int, ...], tuple[float, ...]] = {}

        def evaluate(indices: tuple[int, ...]) -> tuple[float, ...]:
            cached = objective_cache.get(indices)
            if cached is not None:
                return cached
            selected = np.asarray(indices, dtype=int)
            upper = np.triu_indices(len(selected), k=1)
            ideology = self.ideology_matrix[np.ix_(selected, selected)][upper]
            familiar = self.familiarity_matrix[np.ix_(selected, selected)][upper]
            objectives = (
                float(ideology.mean()) if len(ideology) else 0.0,
                float(expertise[selected].mean()),
                0.5
                * (
                    float(self.high_alignment[selected].any())
                    + float(self.mid_alignment[selected].any())
                ),
                float(familiar.mean()) if len(familiar) else 0.0,
            )
            objective_cache[indices] = objectives
            return objectives

        if self.k == self.n_candidates:
            selected = tuple(range(self.n_candidates))
            return selected, evaluate(selected), 1, 0.0

        rng = np.random.RandomState(seed)
        target_size = min(self.population_size, math.comb(self.n_candidates, self.k))

        def random_individual() -> tuple[int, ...]:
            return tuple(
                sorted(rng.choice(self.n_candidates, size=self.k, replace=False).tolist())
            )

        population = list(dict.fromkeys(random_individual() for _ in range(target_size * 2)))
        while len(population) < target_size:
            population.append(random_individual())
            population = list(dict.fromkeys(population))
        population = population[:target_size]

        def score_all(items: list[tuple[int, ...]]) -> list[tuple[float, ...]]:
            return [evaluate(item) for item in items]

        for _ in range(self.generations):
            objectives = score_all(population)
            fronts = nondominated_sort(objectives)
            rank = {
                index: number
                for number, front in enumerate(fronts)
                for index in front
            }
            crowd = {
                index: value
                for front in fronts
                for index, value in crowding_distance(front, objectives).items()
            }

            def tournament() -> tuple[int, ...]:
                left, right = rng.choice(len(population), size=2, replace=False)
                left_key = (rank[left], -crowd[left])
                right_key = (rank[right], -crowd[right])
                return population[left if left_key < right_key else right]

            offspring = []
            while len(offspring) < target_size:
                left, right = tournament(), tournament()
                crossover_pool = sorted(set(left) | set(right))
                child = tuple(
                    sorted(rng.choice(crossover_pool, size=self.k, replace=False).tolist())
                )
                if rng.rand() < 0.25:
                    selected_set = set(child)
                    remove = int(rng.choice(sorted(selected_set)))
                    available = sorted(set(range(self.n_candidates)) - selected_set)
                    selected_set.remove(remove)
                    selected_set.add(int(rng.choice(available)))
                    child = tuple(sorted(selected_set))
                offspring.append(child)

            combined = list(dict.fromkeys(population + offspring))
            combined_objectives = score_all(combined)
            next_population = []
            for front in nondominated_sort(combined_objectives):
                if len(next_population) + len(front) <= target_size:
                    next_population.extend(combined[index] for index in front)
                    continue
                distances = crowding_distance(front, combined_objectives)
                ordered = sorted(front, key=lambda index: distances[index], reverse=True)
                slots = target_size - len(next_population)
                next_population.extend(combined[index] for index in ordered[:slots])
                break
            population = next_population

        objectives = score_all(population)
        first_front = nondominated_sort(objectives)[0]
        values = np.asarray(objectives, dtype=float)
        low = values.min(axis=0)
        span = values.max(axis=0) - low
        normalized = np.divide(
            values - low,
            span,
            out=np.ones_like(values),
            where=span > 0,
        )
        ideal_distances = np.sqrt(np.square(1.0 - normalized).mean(axis=1))
        best = min(
            first_front,
            key=lambda index: (
                float(ideal_distances[index]),
                -float(normalized[index].sum()),
                population[index],
            ),
        )
        return (
            population[best],
            objectives[best],
            len(first_front),
            float(ideal_distances[best]),
        )
