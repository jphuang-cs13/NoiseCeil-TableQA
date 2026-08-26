"""Dependency-free agreement statistics and question-cluster bootstrap.

These utilities contain no benchmark scoring code. They implement the agreement
and question-cluster bootstrap methods used by the released validation workflow.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Callable, Hashable, Mapping, Sequence, TypeVar


Record = TypeVar("Record")


@dataclass(frozen=True)
class BinaryConfusionMatrix:
    judge_0_official_0: int
    judge_0_official_1: int
    judge_1_official_0: int
    judge_1_official_1: int

    @property
    def n(self) -> int:
        return sum(asdict(self).values())


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float | None
    confidence_level: float
    ci_method: str
    lower: float | None
    upper: float | None
    iterations: int
    valid_replicates: int
    invalid_replicates: int
    seed: int


def _require_equal_nonempty(left: Sequence[object], right: Sequence[object]) -> None:
    if len(left) != len(right):
        raise ValueError(f"input length mismatch: {len(left)} != {len(right)}")
    if not left:
        raise ValueError("inputs must not be empty")


def _require_binary(values: Sequence[int], name: str) -> None:
    invalid = [value for value in values if type(value) is not int or value not in (0, 1)]
    if invalid:
        raise ValueError(f"{name} must contain only integer 0/1 labels")


def confusion_matrix(
    judge_labels: Sequence[int], official_labels: Sequence[int]
) -> BinaryConfusionMatrix:
    _require_equal_nonempty(judge_labels, official_labels)
    _require_binary(judge_labels, "judge_labels")
    _require_binary(official_labels, "official_labels")
    counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for judge, official in zip(judge_labels, official_labels):
        counts[(judge, official)] += 1
    return BinaryConfusionMatrix(
        judge_0_official_0=counts[(0, 0)],
        judge_0_official_1=counts[(0, 1)],
        judge_1_official_0=counts[(1, 0)],
        judge_1_official_1=counts[(1, 1)],
    )


def percent_agreement(judge_labels: Sequence[int], official_labels: Sequence[int]) -> float:
    matrix = confusion_matrix(judge_labels, official_labels)
    return (matrix.judge_0_official_0 + matrix.judge_1_official_1) / matrix.n


def cohens_kappa(
    judge_labels: Sequence[int], official_labels: Sequence[int]
) -> float | None:
    matrix = confusion_matrix(judge_labels, official_labels)
    n = matrix.n
    observed = (matrix.judge_0_official_0 + matrix.judge_1_official_1) / n
    judge_0 = matrix.judge_0_official_0 + matrix.judge_0_official_1
    judge_1 = matrix.judge_1_official_0 + matrix.judge_1_official_1
    official_0 = matrix.judge_0_official_0 + matrix.judge_1_official_0
    official_1 = matrix.judge_0_official_1 + matrix.judge_1_official_1
    expected = (judge_0 * official_0 + judge_1 * official_1) / (n * n)
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return None
    return (observed - expected) / denominator


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    _require_equal_nonempty(left, right)
    if len(left) < 2:
        return None
    if not all(math.isfinite(value) for value in (*left, *right)):
        return None
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_ss = math.fsum(value * value for value in left_centered)
    right_ss = math.fsum(value * value for value in right_centered)
    if math.isclose(left_ss, 0.0, abs_tol=1e-30) or math.isclose(
        right_ss, 0.0, abs_tol=1e-30
    ):
        return None
    covariance = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    )
    return covariance / math.sqrt(left_ss * right_ss)


def average_ranks(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("rank input must not be empty")
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = average
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    _require_equal_nonempty(left, right)
    if not all(math.isfinite(value) for value in (*left, *right)):
        return None
    return pearson_correlation(average_ranks(left), average_ranks(right))


def absolute_accuracy_difference_percentage_points(
    judge_accuracies: Sequence[float], official_accuracies: Sequence[float]
) -> dict[str, float]:
    _require_equal_nonempty(judge_accuracies, official_accuracies)
    differences = [
        abs(judge - official) * 100.0
        for judge, official in zip(judge_accuracies, official_accuracies)
    ]
    return {
        "mean_absolute_difference_pp": math.fsum(differences) / len(differences),
        "maximum_absolute_difference_pp": max(differences),
    }


def expand_cluster_sample(
    records: Sequence[Record],
    cluster_key: Callable[[Record], Hashable],
    sampled_clusters: Sequence[Hashable],
) -> list[Record]:
    """Include every row in each sampled cluster, preserving draw/row order."""
    grouped: dict[Hashable, list[Record]] = {}
    for record in records:
        grouped.setdefault(cluster_key(record), []).append(record)
    unknown = [cluster for cluster in sampled_clusters if cluster not in grouped]
    if unknown:
        raise ValueError(f"sampled unknown clusters: {unknown[:3]!r}")
    return [
        record
        for cluster in sampled_clusters
        for record in grouped[cluster]
    ]


def percentile(values: Sequence[float], probability: float) -> float:
    """Linearly interpolated empirical percentile (Hyndman-Fan type 7)."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def question_cluster_bootstrap(
    records: Sequence[Record],
    cluster_key: Callable[[Record], Hashable],
    statistic: Callable[[Sequence[Record]], float | None],
    *,
    iterations: int = 2_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Resample question clusters and keep all repeated experiment rows together."""
    if not records:
        raise ValueError("bootstrap records must not be empty")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    clusters = list(dict.fromkeys(cluster_key(record) for record in records))
    generator = random.Random(seed)
    estimates: list[float] = []
    invalid = 0
    for _ in range(iterations):
        sampled = [clusters[generator.randrange(len(clusters))] for _ in clusters]
        replicate = expand_cluster_sample(records, cluster_key, sampled)
        value = statistic(replicate)
        if value is None or not math.isfinite(value):
            invalid += 1
        else:
            estimates.append(float(value))

    point = statistic(records)
    if point is not None and not math.isfinite(point):
        point = None
    alpha = 1.0 - confidence_level
    lower = percentile(estimates, alpha / 2.0) if estimates else None
    upper = percentile(estimates, 1.0 - alpha / 2.0) if estimates else None
    return BootstrapResult(
        point_estimate=None if point is None else float(point),
        confidence_level=confidence_level,
        ci_method="question-cluster percentile",
        lower=lower,
        upper=upper,
        iterations=iterations,
        valid_replicates=len(estimates),
        invalid_replicates=invalid,
        seed=seed,
    )


def stable_question_identity(record: Mapping[str, object]) -> tuple[str, str, str, str]:
    """Universal identity that remains safe for reused E2E-WTQ query IDs."""
    return (
        str(record["dataset"]),
        str(record["query_id"]),
        str(record["question"]),
        str(record["reference_answer"]),
    )
