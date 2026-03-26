"""
Result storage for benchmark runs.

Stores one JSON file per (scenario, method) pair containing all runs.
Supports incremental/resumable execution.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result from a single optimization run."""

    # Identity
    scenario_name: str
    method_name: str
    start_index: int
    seed: int

    # Parameters
    initial_params: dict[str, float]
    final_params: dict[str, float]
    ground_truth_params: dict[str, float] | None = None  # None for experimental

    # Metrics
    final_loss: float = float("nan")
    best_loss: float = float("nan")
    gamma: float = float("nan")
    n_spikes_model: int = 0
    n_spikes_target: int = 0
    spike_count_error: int = 0
    first_spike_error_ms: float = float("nan")

    # Per-parameter recovery (synthetic only)
    param_recovery: dict[str, dict[str, float]] = field(default_factory=dict)

    # Timing
    wall_time_s: float = 0.0
    n_function_evals: int = 0
    time_per_eval_ms: float = 0.0

    # Diagnostics
    loss_history: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        d = asdict(self)
        # Replace NaN/Inf with None for JSON compatibility
        return _sanitize_for_json(d)

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        """Deserialize from a dictionary."""
        d = _restore_from_json(d)
        return cls(**d)


@dataclass
class ScenarioResults:
    """All runs for one (scenario, method) pair."""

    scenario_name: str
    method_name: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def best_run(self) -> RunResult | None:
        """Run with the highest gamma (or lowest loss if no gamma)."""
        if not self.runs:
            return None
        valid = [r for r in self.runs if not _is_nan(r.gamma)]
        if valid:
            return max(valid, key=lambda r: r.gamma)
        return min(self.runs, key=lambda r: r.final_loss)

    @property
    def mean_gamma(self) -> float:
        """Mean gamma across all runs."""
        gammas = [r.gamma for r in self.runs if not _is_nan(r.gamma)]
        return sum(gammas) / len(gammas) if gammas else float("nan")

    @property
    def best_gamma(self) -> float:
        """Best (highest) gamma across all runs."""
        gammas = [r.gamma for r in self.runs if not _is_nan(r.gamma)]
        return max(gammas) if gammas else float("nan")

    @property
    def n_completed(self) -> int:
        """Number of completed runs."""
        return len(self.runs)

    def summary_dict(self) -> dict:
        """Flat summary for aggregation into tables."""
        import math

        gammas = [r.gamma for r in self.runs if not _is_nan(r.gamma)]
        losses = [r.final_loss for r in self.runs if r.final_loss is not None and not _is_nan(r.final_loss)]
        times = [r.wall_time_s for r in self.runs]

        return {
            "scenario": self.scenario_name,
            "method": self.method_name,
            "n_runs": len(self.runs),
            "mean_gamma": sum(gammas) / len(gammas) if gammas else float("nan"),
            "best_gamma": max(gammas) if gammas else float("nan"),
            "std_gamma": (
                math.sqrt(
                    sum((g - sum(gammas) / len(gammas)) ** 2 for g in gammas)
                    / len(gammas)
                )
                if len(gammas) > 1
                else 0.0
            ),
            "mean_loss": sum(losses) / len(losses) if losses else float("nan"),
            "best_loss": min(losses) if losses else float("nan"),
            "mean_wall_time_s": sum(times) / len(times) if times else 0.0,
            "total_wall_time_s": sum(times),
        }

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "scenario_name": self.scenario_name,
            "method_name": self.method_name,
            "runs": [r.to_dict() for r in self.runs],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioResults":
        """Deserialize from dict."""
        return cls(
            scenario_name=d["scenario_name"],
            method_name=d["method_name"],
            runs=[RunResult.from_dict(r) for r in d["runs"]],
        )


# ── I/O ──────────────────────────────────────────────────────────────────────


def _result_filename(scenario_name: str, method_name: str) -> str:
    """Generate filename for a (scenario, method) pair."""
    return f"{scenario_name}__{method_name}.json"


def save_results(results: ScenarioResults, output_dir: str | Path) -> Path:
    """Save ScenarioResults to a JSON file.

    Args:
        results: Results to save.
        output_dir: Directory to write to.

    Returns:
        Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _result_filename(results.scenario_name, results.method_name)
    with open(path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    log.info("Saved %d runs to %s", len(results.runs), path)
    return path


def load_results(path: str | Path) -> ScenarioResults:
    """Load ScenarioResults from a JSON file."""
    with open(path) as f:
        return ScenarioResults.from_dict(json.load(f))


def load_all_results(results_dir: str | Path) -> list[ScenarioResults]:
    """Load all result files from a directory."""
    results_dir = Path(results_dir)
    results = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            results.append(load_results(path))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning("Skipping %s: %s", path, e)
    return results


# ── JSON helpers ─────────────────────────────────────────────────────────────


def _is_nan(value) -> bool:
    """Check if a value is NaN (works for float and other types)."""
    try:
        import math

        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def _sanitize_for_json(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    import math

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _restore_from_json(obj):
    """Restore None back to NaN for float fields after JSON deserialization."""
    if isinstance(obj, dict):
        restored = {}
        for k, v in obj.items():
            restored[k] = _restore_from_json(v)
        return restored
    if isinstance(obj, list):
        return [_restore_from_json(v) for v in obj]
    return obj
