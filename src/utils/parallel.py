"""Bounded subprocess execution shared by pipeline entry points.

The project uses subprocesses instead of Python threads for model work so that
CPU-bound code is not constrained by the GIL and native numerical libraries can
be given an explicit thread budget.  Keeping scheduling here also prevents each
method runner from inventing a different failure policy.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from typing import Callable, Mapping, Sequence

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class CommandJob:
    """One isolated command and its optional environment overrides."""

    name: str
    command: list[str]
    environment: Mapping[str, str] | None = None
    resource: str = "cpu"


@dataclass(frozen=True)
class CommandResult:
    """Serializable execution record for manifests and summaries."""

    name: str
    command: list[str]
    started_at_utc: str
    ended_at_utc: str
    duration_seconds: float
    return_code: int | None
    status: str
    environment: Mapping[str, str] | None
    resource: str


def resolve_worker_count(n_workers: int) -> int:
    """Resolve ``0`` as an affinity-aware automatic worker count."""

    if n_workers < 0:
        raise ValueError("n_workers must be non-negative (0 means auto)")
    if n_workers:
        return n_workers
    affinity = getattr(os, "sched_getaffinity", None)
    available = len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)
    # Model jobs are generally memory-bandwidth heavy.  A conservative automatic
    # default avoids turning a large server into 128 oversubscribed BLAS jobs.
    return max(1, min(16, available // 4 or 1))


def worker_environment(
    threads_per_worker: int,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment with a bounded native-library thread budget."""

    if threads_per_worker < 1:
        raise ValueError("threads_per_worker must be at least 1")
    environment = os.environ.copy()
    value = str(threads_per_worker)
    for name in THREAD_ENVIRONMENT_VARIABLES:
        environment[name] = value
    if overrides:
        environment.update({str(key): str(item) for key, item in overrides.items()})
    return environment


def _run_one(
    job: CommandJob,
    *,
    cwd: Path,
    threads_per_worker: int,
    dry_run: bool,
    announce: Callable[[str], None] | None,
) -> CommandResult:
    started = datetime.now(timezone.utc)
    return_code: int | None = None
    if not dry_run:
        if announce is not None:
            announce(job.name)
        return_code = subprocess.run(
            job.command,
            cwd=cwd,
            check=False,
            env=worker_environment(threads_per_worker, job.environment),
        ).returncode
    ended = datetime.now(timezone.utc)
    return CommandResult(
        name=job.name,
        command=job.command,
        started_at_utc=started.isoformat(),
        ended_at_utc=ended.isoformat(),
        duration_seconds=(ended - started).total_seconds(),
        return_code=return_code,
        status="dry-run" if dry_run else "ok" if return_code == 0 else "failed",
        environment=job.environment,
        resource=job.resource,
    )


def run_command_jobs(
    jobs: Sequence[CommandJob],
    *,
    cwd: Path,
    n_workers: int = 1,
    threads_per_worker: int = 1,
    dry_run: bool = False,
    continue_on_error: bool = False,
    announce: Callable[[str], None] | None = None,
    resource_limits: Mapping[str, int] | None = None,
) -> list[CommandResult]:
    """Run commands with bounded concurrency and deterministic result ordering."""

    workers = min(resolve_worker_count(n_workers), max(len(jobs), 1))
    if dry_run or workers == 1:
        results: list[CommandResult] = []
        for job in jobs:
            result = _run_one(
                job,
                cwd=cwd,
                threads_per_worker=threads_per_worker,
                dry_run=dry_run,
                announce=announce,
            )
            results.append(result)
            if result.return_code not in (None, 0) and not continue_on_error:
                break
        return results

    indexed_results: dict[int, CommandResult] = {}
    pending = list(range(len(jobs)))
    stop_submitting = False
    futures: dict[Future[CommandResult], int] = {}
    limits = {str(key): int(value) for key, value in (resource_limits or {}).items()}
    if any(value < 1 for value in limits.values()):
        raise ValueError("Every resource limit must be at least 1")
    resources_in_use: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="method-job") as executor:
        while pending or futures:
            submitted = True
            while not stop_submitting and pending and len(futures) < workers and submitted:
                submitted = False
                for pending_position, index in enumerate(pending):
                    job = jobs[index]
                    limit = limits.get(job.resource, workers)
                    if resources_in_use.get(job.resource, 0) >= limit:
                        continue
                    pending.pop(pending_position)
                    resources_in_use[job.resource] = resources_in_use.get(job.resource, 0) + 1
                    submitted = True
                    break
                if not submitted:
                    break
                future = executor.submit(
                    _run_one,
                    job,
                    cwd=cwd,
                    threads_per_worker=threads_per_worker,
                    dry_run=False,
                    announce=announce,
                )
                futures[future] = index
            if not futures:
                break
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                index = futures.pop(future)
                result = future.result()
                indexed_results[index] = result
                resource = jobs[index].resource
                resources_in_use[resource] -= 1
                if result.return_code not in (None, 0) and not continue_on_error:
                    stop_submitting = True
    return [indexed_results[index] for index in sorted(indexed_results)]
