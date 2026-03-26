"""
Result analysis utilities for benchmark results.

Provides summary tables, per-parameter recovery analysis,
and LaTeX table generation for thesis.
"""

import math


def summary_table(results: list) -> list[dict]:
    """Generate a flat summary table from ScenarioResults.

    Args:
        results: List of ScenarioResults objects.

    Returns:
        List of flat dicts suitable for pandas DataFrame or tabulation.
    """
    return [r.summary_dict() for r in results]


def recovery_table(results: list, param_names: list[str]) -> list[dict]:
    """Generate per-parameter recovery statistics.

    Only meaningful for synthetic scenarios with ground truth.

    Args:
        results: List of ScenarioResults objects.
        param_names: Parameter names to include.

    Returns:
        List of dicts with per-parameter recovery stats.
    """
    rows = []
    for sr in results:
        for param in param_names:
            errors_pct = []
            errors_abs = []
            for run in sr.runs:
                if param in run.param_recovery:
                    rec = run.param_recovery[param]
                    if not _is_nan(rec.get("error_pct", float("nan"))):
                        errors_pct.append(rec["error_pct"])
                    if not _is_nan(rec.get("error_abs", float("nan"))):
                        errors_abs.append(rec["error_abs"])

            if not errors_pct:
                continue

            rows.append(
                {
                    "scenario": sr.scenario_name,
                    "method": sr.method_name,
                    "parameter": param,
                    "n_runs": len(errors_pct),
                    "mean_error_pct": sum(errors_pct) / len(errors_pct),
                    "best_error_pct": min(errors_pct),
                    "std_error_pct": (
                        math.sqrt(
                            sum(
                                (e - sum(errors_pct) / len(errors_pct)) ** 2
                                for e in errors_pct
                            )
                            / len(errors_pct)
                        )
                        if len(errors_pct) > 1
                        else 0.0
                    ),
                    "mean_error_abs": sum(errors_abs) / len(errors_abs),
                    "best_error_abs": min(errors_abs),
                }
            )
    return rows


def print_table(rows: list[dict], max_width: int = 120) -> None:
    """Print a formatted table to console.

    Args:
        rows: List of flat dicts (all with same keys).
        max_width: Maximum table width.
    """
    if not rows:
        print("(no results)")
        return

    headers = list(rows[0].keys())

    # Format values
    formatted = []
    for row in rows:
        fmt_row = {}
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                if _is_nan(val):
                    fmt_row[h] = "NaN"
                else:
                    fmt_row[h] = f"{val:.4f}"
            else:
                fmt_row[h] = str(val)
        formatted.append(fmt_row)

    # Compute column widths
    widths = {h: len(h) for h in headers}
    for row in formatted:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))

    # Print header
    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    print(header_line[:max_width])
    print("-" * min(len(header_line), max_width))

    # Print rows
    for row in formatted:
        line = " | ".join(row[h].ljust(widths[h]) for h in headers)
        print(line[:max_width])


def format_latex_table(rows: list[dict], caption: str = "") -> str:
    """Generate a LaTeX tabular from result rows.

    Args:
        rows: List of flat dicts.
        caption: Table caption.

    Returns:
        LaTeX table string.
    """
    if not rows:
        return "% No results"

    headers = list(rows[0].keys())
    n_cols = len(headers)
    col_spec = "l" + "r" * (n_cols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}" if caption else "",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        cells = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                if _is_nan(val):
                    cells.append("--")
                else:
                    cells.append(f"{val:.3f}")
            else:
                cells.append(_latex_escape(str(val)))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(line for line in lines if line)


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    for char in ["_", "%", "&", "#", "$"]:
        text = text.replace(char, f"\\{char}")
    return text


def _is_nan(value) -> bool:
    """Check if a value is NaN."""
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False
