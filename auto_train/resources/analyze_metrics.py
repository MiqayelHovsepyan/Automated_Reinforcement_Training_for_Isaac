# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extract TensorBoard metrics from a training run and output as JSON.

Includes convergence detection, curve shape analysis, and suspicious pattern
detection for automated RL training quality assessment.

Usage:
    python .claude/skills/auto_train/resources/analyze_metrics.py --log-dir <path> [--output <path>/metrics.json]
"""

import argparse
import json
import os
import sys

import numpy as np
from tbparse import SummaryReader


def compute_trend(values: list[float], fraction: float = 0.2) -> str:
    """Compute trend from the last `fraction` of values using linear regression slope."""
    if len(values) < 10:
        return "insufficient_data"
    n = max(int(len(values) * fraction), 5)
    tail = values[-n:]
    x = np.arange(len(tail), dtype=np.float64)
    y = np.array(tail, dtype=np.float64)
    # Linear regression: slope = cov(x,y) / var(x)
    slope = np.cov(x, y)[0, 1] / (np.var(x) + 1e-12)
    # Normalize slope relative to the mean magnitude
    mean_abs = np.mean(np.abs(y)) + 1e-12
    normalized_slope = slope / mean_abs
    if normalized_slope > 0.01:
        return "improving"
    elif normalized_slope < -0.01:
        return "degrading"
    return "stable"


def detect_convergence(
    values: list[float],
    steps: list[int] | None = None,
    window_size: int = 50,
    min_stable_windows: int = 5,
    relative_delta: float = 0.02,
) -> dict:
    """Detect where a metric converged (stopped improving significantly).

    Uses a sliding window running mean. Convergence is the first point where
    `min_stable_windows` consecutive windows all have relative change < `relative_delta`.

    Returns:
        dict with: converged, converged_at_index, converged_at_iteration,
        converged_value, percent_of_training
    """
    result = {
        "converged": False,
        "converged_at_index": None,
        "converged_at_iteration": None,
        "converged_value": None,
        "percent_of_training": None,
    }

    if len(values) < window_size * 2:
        return result

    arr = np.array(values, dtype=np.float64)

    # Compute running mean
    kernel = np.ones(window_size) / window_size
    running_mean = np.convolve(arr, kernel, mode="valid")

    if len(running_mean) < min_stable_windows + 1:
        return result

    # Compute relative change between consecutive running means
    diffs = np.abs(np.diff(running_mean))
    magnitudes = np.abs(running_mean[1:]) + 1e-12
    relative_changes = diffs / magnitudes

    # Find first index where min_stable_windows consecutive changes are all < relative_delta
    consecutive = 0
    for i, rc in enumerate(relative_changes):
        if rc < relative_delta:
            consecutive += 1
            if consecutive >= min_stable_windows:
                # Convergence found — map back to original array index
                conv_index = i - min_stable_windows + 1 + window_size - 1
                conv_index = min(conv_index, len(values) - 1)

                result["converged"] = True
                result["converged_at_index"] = int(conv_index)
                result["converged_value"] = float(running_mean[i - min_stable_windows + 1])
                result["percent_of_training"] = round(conv_index / len(values) * 100, 1)

                if steps and conv_index < len(steps):
                    result["converged_at_iteration"] = int(steps[conv_index])

                return result
        else:
            consecutive = 0

    return result


def analyze_curve_shape(values: list[float], convergence_info: dict) -> str:
    """Classify the overall shape of a training curve.

    Returns one of:
        "still_improving" — slope positive in last 10%
        "converged" — plateaued and stable
        "converged_early" — plateaued before 50% of training
        "degrading" — getting worse
        "oscillating" — high variance, not converging
        "insufficient_data" — too few data points
    """
    if len(values) < 10:
        return "insufficient_data"

    arr = np.array(values, dtype=np.float64)

    # Compute last-10% normalized slope
    n_tail = max(int(len(arr) * 0.1), 5)
    tail = arr[-n_tail:]
    x = np.arange(len(tail), dtype=np.float64)
    slope = np.cov(x, tail)[0, 1] / (np.var(x) + 1e-12)
    mean_abs = np.mean(np.abs(tail)) + 1e-12
    normalized_slope = slope / mean_abs

    # Check for degrading
    if normalized_slope < -0.01:
        return "degrading"

    # Check for oscillating — coefficient of variation of last 20%
    n_var = max(int(len(arr) * 0.2), 5)
    var_tail = arr[-n_var:]
    mean_val = np.mean(var_tail)
    if abs(mean_val) > 1e-12:
        cov = np.std(var_tail) / abs(mean_val)
        if cov > 0.15:
            return "oscillating"

    # Check for early convergence
    if convergence_info.get("converged") and convergence_info.get("percent_of_training", 100) < 50.0:
        return "converged_early"

    # Check for convergence
    if convergence_info.get("converged"):
        return "converged"

    # Check if still improving
    if normalized_slope > 0.01:
        return "still_improving"

    # Near-zero slope, low variance — effectively converged
    return "converged"


def detect_suspicious_patterns(
    reward_terms: dict[str, dict],
    scalars: dict[str, dict],
) -> list[dict]:
    """Detect reward hacking and other suspicious training patterns.

    Checks for:
    1. Gait reward gaming velocity tracking
    2. High total reward with low tracking rewards
    3. Tracking terms that flatlined early
    4. Non-zero base/body contact terminations

    Returns:
        List of dicts with: pattern, severity ("warning"|"critical"), details
    """
    patterns = []

    # Identify tracking and gait terms by name
    tracking_keywords = ("track_lin_vel", "track_ang_vel", "linear_velocity", "angular_velocity")
    gait_keywords = ("gait",)

    tracking_terms = {
        k: v for k, v in reward_terms.items() if any(kw in k.lower() for kw in tracking_keywords)
    }
    gait_terms = {
        k: v for k, v in reward_terms.items() if any(kw in k.lower() for kw in gait_keywords)
    }

    # Pattern 1: Gait gaming velocity tracking
    for gname, gdata in gait_terms.items():
        g_shape = gdata.get("curve_shape", "")
        g_final = gdata.get("final", 0) or 0
        if g_shape in ("still_improving", "converged") and g_final > 0:
            for tname, tdata in tracking_terms.items():
                t_shape = tdata.get("curve_shape", "")
                t_final = tdata.get("final", 0) or 0
                t_conv_iter = tdata.get("converged_at_iteration")
                if t_shape == "converged_early":
                    patterns.append({
                        "pattern": "gait_gaming_tracking",
                        "severity": "critical",
                        "details": (
                            f"'{gname}' is strong (final={g_final:.3f}, shape={g_shape}) but "
                            f"'{tname}' converged early at iter {t_conv_iter} "
                            f"with low value {t_final:.3f}"
                        ),
                    })

    # Pattern 2: High total reward but low tracking contribution
    mean_reward_data = None
    for key in ("Train/mean_reward", "Train/reward"):
        if key in scalars:
            mean_reward_data = scalars[key]
            break

    if mean_reward_data and tracking_terms:
        total_reward = mean_reward_data.get("final", 0) or 0
        tracking_sum = sum((v.get("final", 0) or 0) for v in tracking_terms.values())
        if total_reward > 0 and tracking_sum >= 0:
            tracking_ratio = tracking_sum / (total_reward + 1e-12)
            if tracking_ratio < 0.15:
                patterns.append({
                    "pattern": "high_reward_low_tracking",
                    "severity": "critical",
                    "details": (
                        f"Total reward is {total_reward:.2f} but tracking terms sum to "
                        f"{tracking_sum:.3f} ({tracking_ratio*100:.1f}% of total). "
                        f"Robot may be exploiting non-tracking rewards."
                    ),
                })

    # Pattern 3: Tracking terms flatlined early
    for tname, tdata in tracking_terms.items():
        t_shape = tdata.get("curve_shape", "")
        t_final = tdata.get("final", 0) or 0
        t_conv_iter = tdata.get("converged_at_iteration")
        if t_shape == "converged_early":
            patterns.append({
                "pattern": "tracking_flatlined_early",
                "severity": "warning",
                "details": (
                    f"'{tname}' converged early (iter {t_conv_iter}) "
                    f"at value {t_final:.3f} — may not be learning to track velocity."
                ),
            })

    # Pattern 4: Body contact terminations non-zero
    contact_keys = [k for k in scalars if "base_contact" in k.lower() or "body_contact" in k.lower()]
    for key in contact_keys:
        mean_val = scalars[key].get("mean_last_100", 0) or 0
        if mean_val > 0.01:
            patterns.append({
                "pattern": "body_contact_nonzero",
                "severity": "warning",
                "details": f"'{key}' mean_last_100 = {mean_val:.3f} — robot body is touching ground.",
            })

    return patterns


def analyze_scalar(values: list[float], steps: list[int] | None = None) -> dict:
    """Compute summary statistics, convergence, and curve shape for a scalar time series."""
    if not values:
        return {
            "final": None,
            "mean_last_100": None,
            "std_last_100": None,
            "max": None,
            "min": None,
            "trend": "no_data",
            "num_points": 0,
            "curve_shape": "insufficient_data",
            "converged_at_iteration": None,
            "converged_value": None,
            "percent_of_training_at_convergence": None,
        }

    arr = np.array(values, dtype=np.float64)
    last_100 = arr[-100:] if len(arr) >= 100 else arr

    convergence_info = detect_convergence(values, steps)
    curve_shape = analyze_curve_shape(values, convergence_info)

    return {
        "final": float(arr[-1]),
        "mean_last_100": float(np.mean(last_100)),
        "std_last_100": float(np.std(last_100)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "trend": compute_trend(values),
        "num_points": len(values),
        "curve_shape": curve_shape,
        "converged_at_iteration": convergence_info["converged_at_iteration"],
        "converged_value": convergence_info["converged_value"],
        "percent_of_training_at_convergence": convergence_info["percent_of_training"],
    }


def main():
    parser = argparse.ArgumentParser(description="Extract TensorBoard metrics to JSON.")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to training run log directory.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path. Defaults to <log-dir>/metrics.json.")
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    output_path = args.output or os.path.join(log_dir, "metrics.json")

    if not os.path.isdir(log_dir):
        print(f"[ERROR] Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    # Read all scalar events
    try:
        reader = SummaryReader(log_dir)
        df = reader.scalars
    except Exception as e:
        print(f"[ERROR] Failed to read TensorBoard events: {e}", file=sys.stderr)
        # Write empty report
        report = {"log_dir": log_dir, "error": str(e), "scalars": {}, "reward_terms": {}, "suspicious_patterns": []}
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(1)

    if df.empty:
        print("[WARNING] No scalar data found in TensorBoard events.", file=sys.stderr)
        report = {
            "log_dir": log_dir,
            "total_iterations": 0,
            "scalars": {},
            "reward_terms": {},
            "suspicious_patterns": [],
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        return

    # Get all unique tags
    tags = df["tag"].unique().tolist()

    # Separate main scalars from reward terms
    scalars = {}
    reward_terms = {}

    for tag in tags:
        tag_df = df[df["tag"] == tag].sort_values("step")
        values = tag_df["value"].tolist()
        steps = tag_df["step"].tolist()
        summary = analyze_scalar(values, steps)

        if tag.startswith("Episode_Reward/") or tag.startswith("Reward/"):
            # Individual reward term (RSL-RL logs as Episode_Reward/<name>)
            term_name = tag.split("/", 1)[1]
            reward_terms[term_name] = summary
        else:
            scalars[tag] = summary

    # Detect suspicious patterns
    suspicious_patterns = detect_suspicious_patterns(reward_terms, scalars)

    # Compute total iterations from max step
    total_iterations = int(df["step"].max()) if not df.empty else 0

    # Compute wall time
    wall_time_seconds = None
    if "wall_time" in df.columns and not df["wall_time"].isna().all():
        wall_time_seconds = float(df["wall_time"].max() - df["wall_time"].min())

    report = {
        "log_dir": log_dir,
        "total_iterations": total_iterations,
        "wall_time_seconds": wall_time_seconds,
        "scalars": scalars,
        "reward_terms": reward_terms,
        "suspicious_patterns": suspicious_patterns,
        "all_tags": tags,
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[INFO] Metrics report written to: {output_path}")
    print(f"[INFO] Total iterations: {total_iterations}, Tags found: {len(tags)}")
    if reward_terms:
        print(f"[INFO] Reward terms: {', '.join(sorted(reward_terms.keys()))}")

    # Print convergence warnings
    for term_name, term_data in reward_terms.items():
        shape = term_data.get("curve_shape", "?")
        conv_iter = term_data.get("converged_at_iteration")
        if shape in ("converged_early", "degrading", "oscillating"):
            msg = f"[WARNING] Reward term '{term_name}': {shape}"
            if conv_iter is not None:
                msg += f" at iter {conv_iter}"
            print(msg)

    for key, sdata in scalars.items():
        shape = sdata.get("curve_shape", "?")
        conv_iter = sdata.get("converged_at_iteration")
        if shape in ("converged_early", "degrading", "oscillating"):
            msg = f"[WARNING] Scalar '{key}': {shape}"
            if conv_iter is not None:
                msg += f" at iter {conv_iter}"
            print(msg)

    # Print suspicious patterns
    if suspicious_patterns:
        print(f"\n[WARNING] {len(suspicious_patterns)} suspicious pattern(s) detected:")
        for sp in suspicious_patterns:
            print(f"  [{sp['severity'].upper()}] {sp['pattern']}: {sp['details']}")


if __name__ == "__main__":
    main()
