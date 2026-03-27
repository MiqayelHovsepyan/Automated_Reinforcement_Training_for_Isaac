# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Orchestrate a single auto-training phase: train → metrics → play → frames → report.

This script runs as a normal Python process (no Isaac Sim dependency). It launches
train_with_overrides.py and play_for_inspection.py as subprocesses and coordinates the pipeline.

Usage:
    python .claude/skills/auto_train/resources/run_phase.py \
        --task Isaac-Velocity-Spot-Like-Flat-Ayg-v0 \
        --play-task Isaac-Velocity-Spot-Like-Flat-Ayg-Play-v0 \
        --overrides-file .claude/skills/auto_train/experiments/.scratch/phase_0_overrides.json \
        --max-iterations 2000 --num-envs 4096 --headless \
        --monitor-interval 100 --abort-plateau-patience 300
"""

import argparse
import glob
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time


def find_newest_log_dir(logs_base: str) -> str | None:
    """Find the most recently created log directory under logs_base."""
    if not os.path.isdir(logs_base):
        return None
    subdirs = []
    for entry in os.scandir(logs_base):
        if entry.is_dir():
            subdirs.append(entry.path)
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)


def find_log_dir_from_output(output: str) -> str | None:
    """Parse the training subprocess output for the log directory marker."""
    for line in output.splitlines():
        if "[AUTO_TRAIN_LOG_DIR]" in line:
            return line.split("[AUTO_TRAIN_LOG_DIR]")[-1].strip()
    return None


def find_latest_checkpoint(log_dir: str) -> str | None:
    """Find the latest model checkpoint in a log directory."""
    pattern = os.path.join(log_dir, "model_*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    # Sort by iteration number extracted from filename
    def extract_iter(path):
        match = re.search(r"model_(\d+)\.pt", os.path.basename(path))
        return int(match.group(1)) if match else -1
    return max(checkpoints, key=extract_iter)


def find_play_video(log_dir: str) -> str | None:
    """Find the play video MP4 in a log directory."""
    video_dir = os.path.join(log_dir, "videos", "play")
    if not os.path.isdir(video_dir):
        return None
    mp4s = glob.glob(os.path.join(video_dir, "*.mp4"))
    return mp4s[0] if mp4s else None


def read_metrics_for_monitoring(log_dir: str) -> dict | None:
    """Quick read of TB events for monitoring (lightweight, no full analysis)."""
    try:
        from tbparse import SummaryReader
        reader = SummaryReader(log_dir)
        df = reader.scalars
        if df.empty:
            return None
        # Get mean_reward series
        reward_tags = [t for t in df["tag"].unique() if "mean_reward" in t.lower() or "reward" == t.lower().split("/")[-1]]
        if not reward_tags:
            return None
        tag = reward_tags[0]
        tag_df = df[df["tag"] == tag].sort_values("step")
        return {
            "tag": tag,
            "steps": tag_df["step"].tolist(),
            "values": tag_df["value"].tolist(),
            "current_iteration": int(tag_df["step"].max()),
        }
    except Exception:
        return None


def check_abort_criteria(metrics: dict, args) -> str | None:
    """Check if any abort criteria are met. Returns reason string or None."""
    if not metrics or not metrics["values"]:
        return None

    values = metrics["values"]
    steps = metrics["steps"]
    current_iter = metrics["current_iteration"]

    # Check min reward threshold
    if args.abort_min_reward_at:
        parts = args.abort_min_reward_at.split(":")
        if len(parts) == 2:
            threshold_iter, threshold_val = int(parts[0]), float(parts[1])
            if current_iter >= threshold_iter and values[-1] < threshold_val:
                return f"reward {values[-1]:.2f} < {threshold_val} after {current_iter} iterations"

    # Check plateau — patience is in iterations, so use steps to find the window
    if args.abort_plateau_patience and current_iter > args.abort_plateau_patience:
        patience = args.abort_plateau_patience
        delta = args.abort_plateau_min_delta
        cutoff_step = current_iter - patience
        # Select values whose step falls within the patience window
        recent = [v for s, v in zip(steps, values) if s >= cutoff_step]
        if len(recent) >= 2:
            best_in_window = max(recent)
            worst_in_window = min(recent)
            if (best_in_window - worst_in_window) < delta:
                return f"plateau for {patience} iterations (range {worst_in_window:.2f}-{best_in_window:.2f})"

    # Check reward collapse (significant drop from peak, used as proxy for episode quality)
    if args.abort_episode_length_drop:
        if len(values) > 100:
            max_reward = max(values)
            if max_reward > 0 and values[-1] < max_reward * args.abort_episode_length_drop:
                return f"reward dropped to {values[-1]:.2f} from max {max_reward:.2f} (ratio {values[-1]/max_reward:.2f})"

    return None


# Regex patterns for parsing RSL-RL training stdout
_RE_LEARNING_ITER = re.compile(r"Learning iteration (\d+)/(\d+)")
_RE_MEAN_REWARD = re.compile(r"Mean reward:\s*([-\d.]+)")
_RE_FPS = re.compile(r"Computation:\s*(\d+)\s*steps/s")


def parse_training_line(line: str, progress_state: dict):
    """Parse a training stdout line and update progress_state in-place."""
    m = _RE_LEARNING_ITER.search(line)
    if m:
        with progress_state["_lock"]:
            progress_state["current_iteration"] = int(m.group(1))
            progress_state["max_iterations"] = int(m.group(2))
    m = _RE_MEAN_REWARD.search(line)
    if m:
        with progress_state["_lock"]:
            progress_state["mean_reward"] = float(m.group(1))
    m = _RE_FPS.search(line)
    if m:
        with progress_state["_lock"]:
            progress_state["fps"] = float(m.group(1))


def _write_running_report(report_path: str, task: str, progress_state: dict):
    """Write a running-status report with progress info to the external report path."""
    with progress_state["_lock"]:
        cur_iter = progress_state["current_iteration"]
        max_iter = progress_state["max_iterations"]
        mean_reward = progress_state["mean_reward"]
        fps = progress_state["fps"]
        start_time = progress_state["_start_time"]

    elapsed = time.time() - start_time
    eta = None
    percent = 0.0
    if max_iter > 0 and cur_iter > 0:
        percent = round(cur_iter / max_iter * 100, 1)
        iter_per_sec = cur_iter / elapsed if elapsed > 0 else 0
        if iter_per_sec > 0:
            eta = round((max_iter - cur_iter) / iter_per_sec)

    report = {
        "status": "running",
        "task": task,
        "pid": os.getpid(),
        "progress": {
            "current_iteration": cur_iter,
            "max_iterations": max_iter,
            "percent_complete": percent,
            "mean_reward": mean_reward,
            "eta_seconds": eta,
            "elapsed_seconds": round(elapsed),
            "fps": fps,
        },
    }
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
    except OSError:
        pass  # don't crash if write fails


def _validate_overrides_file(filepath: str) -> str | None:
    """Validate an overrides JSON file. Returns error message or None if valid."""
    if not os.path.isfile(filepath):
        return f"File not found: {filepath}"
    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    if not isinstance(data, dict):
        return f"Expected a JSON object (dict), got {type(data).__name__}"

    errors = []
    for key, value in data.items():
        # Check for common nested-dict mistake: {"env_cfg": {"key": val}}
        if isinstance(value, dict):
            errors.append(
                f"  Key '{key}' has a dict value. Use flat dot-paths instead "
                f"(e.g., \"{key}.subkey\": value, not \"{key}\": {{\"subkey\": value}})"
            )
        # Check for empty keys
        if not key or not key.strip():
            errors.append("  Found empty key in overrides")

    if errors:
        return "Override format errors:\n" + "\n".join(errors)
    return None


def _progress_writer(report_path: str, task: str, progress_state: dict, stop_event: threading.Event, interval: int = 30):
    """Background thread: periodically writes progress to external report file."""
    while not stop_event.wait(timeout=interval):
        _write_running_report(report_path, task, progress_state)


def main():
    parser = argparse.ArgumentParser(description="Run one auto-training phase.")
    # Task configuration
    parser.add_argument("--task", type=str, required=True, help="Training task ID.")
    parser.add_argument("--play-task", type=str, default=None,
                        help="Play task ID. If not provided, derives from --task by adding '-Play'.")
    parser.add_argument("--overrides-file", type=str, default=None, help="JSON overrides file path.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Max training iterations.")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--headless", action="store_true", default=False, help="Run without GUI.")

    # Resume
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a previous run directory to resume from.")

    # Play / video
    parser.add_argument("--video-length", type=int, default=300, help="Play video length in steps.")
    parser.add_argument("--num-frames", type=int, default=12, help="Number of frames to extract from play video.")
    parser.add_argument("--num-play-envs", type=int, default=4,
                        help="Number of environments for play/video recording (default: 4). "
                             "Use 2-4 for clear side-view video of individual robots.")
    parser.add_argument("--skip-play", action="store_true", default=False, help="Skip play and video extraction.")

    # Monitoring / abort criteria
    parser.add_argument("--monitor-interval", type=int, default=0,
                        help="Check abort criteria every N seconds during training. 0 = disabled.")
    parser.add_argument("--abort-plateau-patience", type=int, default=None,
                        help="Abort if reward hasn't improved in this many iterations.")
    parser.add_argument("--abort-plateau-min-delta", type=float, default=0.5,
                        help="Minimum improvement to not be considered a plateau.")
    parser.add_argument("--abort-min-reward-at", type=str, default=None,
                        help="Abort if reward < VALUE after ITER iterations. Format: 'ITER:VALUE'")
    parser.add_argument("--abort-episode-length-drop", type=float, default=None,
                        help="Abort if reward drops below this ratio of max observed.")

    # External report path (for polling by Claude when run detached via nohup)
    parser.add_argument("--report-path", type=str, default=None,
                        help="Write phase report to this path (in addition to inside log_dir). "
                             "A 'running' status is written immediately at launch so callers can poll.")

    args = parser.parse_args()

    # Derive play task if not provided
    if args.play_task is None:
        # Isaac-Velocity-Flat-Ayg-v0 -> Isaac-Velocity-Flat-Ayg-Play-v0
        task = args.task
        if "-v0" in task and "-Play-" not in task:
            args.play_task = task.replace("-v0", "-Play-v0")
        else:
            args.play_task = task
        print(f"[INFO] Derived play task: {args.play_task}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Script lives at .claude/skills/auto_train/resources/ — go up 4 levels to cf_lab root
    cf_lab_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))

    # Shared progress state updated by stdout parser, read by progress writer
    progress_state = {
        "current_iteration": 0,
        "max_iterations": args.max_iterations or 0,
        "mean_reward": None,
        "fps": None,
        "_lock": threading.Lock(),
        "_start_time": time.time(),
    }

    # Write "running" marker so callers can poll for completion
    report_path_ext = None
    if args.report_path:
        report_path_ext = os.path.abspath(args.report_path)
        os.makedirs(os.path.dirname(report_path_ext) or ".", exist_ok=True)
        _write_running_report(report_path_ext, args.task, progress_state)
        print(f"[INFO] Status file: {report_path_ext}")

    start_time = time.time()
    log_dir = None
    status = "completed"
    early_stop_reason = None
    iterations_completed = None  # Will be resolved from metrics after training

    # ── Pre-flight: validate overrides JSON ──
    if args.overrides_file:
        validation_error = _validate_overrides_file(args.overrides_file)
        if validation_error:
            status = "validation_error"
            print(f"[ERROR] Override validation failed: {validation_error}")
            if report_path_ext:
                with open(report_path_ext, "w") as f:
                    json.dump({"status": "validation_error", "error": validation_error, "task": args.task}, f, indent=2)
            sys.exit(1)

    # ── Pre-flight: validate resume path ──
    if args.resume_from:
        # Check if it looks like a full path instead of just a run folder name
        if "/" in args.resume_from and not args.resume_from.startswith("logs"):
            print(f"[WARNING] --resume-from should be just the run folder name (e.g., '2026-03-21_01-52-22'), "
                  f"not a full path. Got: {args.resume_from}")

    # ── Step 1: Train ──
    print("=" * 60)
    print("[PHASE] Step 1: Training")
    print("=" * 60)

    train_cmd = [
        sys.executable, os.path.join(script_dir, "train_with_overrides.py"),
        f"--task={args.task}",
    ]
    if args.overrides_file:
        train_cmd.append(f"--overrides-file={os.path.abspath(args.overrides_file)}")
    if args.max_iterations is not None:
        train_cmd.append(f"--max_iterations={args.max_iterations}")
    if args.num_envs is not None:
        train_cmd.append(f"--num_envs={args.num_envs}")
    if args.seed is not None:
        train_cmd.append(f"--seed={args.seed}")
    if args.headless:
        train_cmd.append("--headless")
    if args.resume_from:
        train_cmd.extend(["--resume", f"--load_run={args.resume_from}"])

    print(f"[INFO] Command: {' '.join(train_cmd)}")

    # Launch training subprocess
    train_proc = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cf_lab_dir,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )

    train_output_lines = []

    def _pipe_reader(pipe, line_queue):
        """Read lines from pipe in a dedicated thread (avoids text-mode buffering issues)."""
        try:
            for line in pipe:
                line_queue.put(line.rstrip())
        except ValueError:
            pass  # pipe closed
        finally:
            line_queue.put(None)  # sentinel: EOF

    # Start progress writer thread if external report path is set
    progress_stop = threading.Event()
    progress_thread = None
    if report_path_ext:
        progress_thread = threading.Thread(
            target=_progress_writer,
            args=(report_path_ext, args.task, progress_state, progress_stop),
            daemon=True,
        )
        progress_thread.start()

    if args.monitor_interval > 0 and (args.abort_plateau_patience or args.abort_min_reward_at or args.abort_episode_length_drop):
        # Monitoring mode: read output via thread, periodically check abort criteria
        print(f"[INFO] Monitoring enabled: checking every {args.monitor_interval}s")
        last_check = time.time()

        line_queue = queue.Queue()
        reader_thread = threading.Thread(target=_pipe_reader, args=(train_proc.stdout, line_queue), daemon=True)
        reader_thread.start()

        eof = False
        while not eof and train_proc.poll() is None:
            # Drain all available lines (non-blocking)
            while True:
                try:
                    line = line_queue.get_nowait()
                    if line is None:
                        eof = True
                        break
                    train_output_lines.append(line)
                    parse_training_line(line, progress_state)
                    print(line)
                except queue.Empty:
                    break

            if eof:
                break

            # Periodic abort check
            if time.time() - last_check >= args.monitor_interval:
                last_check = time.time()
                # Try to find the log dir from output so far
                if log_dir is None:
                    log_dir = find_log_dir_from_output("\n".join(train_output_lines))
                if log_dir:
                    metrics = read_metrics_for_monitoring(log_dir)
                    if metrics:
                        reason = check_abort_criteria(metrics, args)
                        if reason:
                            print(f"\n[ABORT] Early stopping: {reason}")
                            status = "early_stopped"
                            early_stop_reason = reason
                            iterations_completed = metrics["current_iteration"]
                            # Gracefully terminate
                            train_proc.send_signal(signal.SIGTERM)
                            try:
                                train_proc.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                train_proc.kill()
                            break

            time.sleep(1)  # check every 1 second

        # Drain any remaining lines after loop exits
        while True:
            try:
                line = line_queue.get(timeout=2)
                if line is None:
                    break
                train_output_lines.append(line)
                parse_training_line(line, progress_state)
                print(line)
            except queue.Empty:
                break
        reader_thread.join(timeout=5)
    else:
        # No monitoring — stream output and parse progress
        for line in train_proc.stdout:
            stripped = line.rstrip()
            train_output_lines.append(stripped)
            parse_training_line(stripped, progress_state)
            print(stripped)

    train_proc.wait()
    train_output = "\n".join(train_output_lines)

    # Stop progress writer
    progress_stop.set()
    if progress_thread:
        progress_thread.join(timeout=5)

    if train_proc.returncode != 0 and status != "early_stopped":
        status = "failed"
        print(f"[ERROR] Training failed with return code {train_proc.returncode}")

    # Find log directory
    if log_dir is None:
        log_dir = find_log_dir_from_output(train_output)
    if log_dir is None:
        # Fallback: find newest log dir
        logs_base = os.path.join(cf_lab_dir, "logs", "rsl_rl")
        if os.path.isdir(logs_base):
            # Search all experiment subdirs for the newest run
            newest = None
            newest_time = 0
            for exp_dir in os.scandir(logs_base):
                if exp_dir.is_dir():
                    candidate = find_newest_log_dir(exp_dir.path)
                    if candidate and os.path.getmtime(candidate) > newest_time:
                        newest = candidate
                        newest_time = os.path.getmtime(candidate)
            log_dir = newest

    if log_dir is None:
        print("[ERROR] Could not determine log directory. Aborting.")
        sys.exit(1)

    print(f"\n[INFO] Log directory: {log_dir}")
    training_time = time.time() - start_time

    # ── Step 2: Extract metrics ──
    print("\n" + "=" * 60)
    print("[PHASE] Step 2: Extracting metrics")
    print("=" * 60)

    metrics_output = os.path.join(log_dir, "metrics.json")
    metrics_cmd = [
        sys.executable, os.path.join(script_dir, "analyze_metrics.py"),
        f"--log-dir={log_dir}",
        f"--output={metrics_output}",
    ]
    subprocess.run(metrics_cmd, cwd=cf_lab_dir)

    metrics_data = {}
    if os.path.isfile(metrics_output):
        with open(metrics_output) as f:
            metrics_data = json.load(f)

    # Resolve actual iterations completed from metrics (fallback to max_iterations)
    if iterations_completed is None:
        iterations_completed = metrics_data.get("total_iterations", args.max_iterations)

    # ── Step 3: Play + record video ──
    frame_paths = []
    video_frames_dir = None

    if not args.skip_play and status != "failed":
        print("\n" + "=" * 60)
        print("[PHASE] Step 3: Playing policy and recording video")
        print("=" * 60)

        checkpoint = find_latest_checkpoint(log_dir)
        if checkpoint:
            play_cmd = [
                sys.executable, os.path.join(script_dir, "play_for_inspection.py"),
                f"--task={args.play_task}",
                "--video",
                f"--video_length={args.video_length}",
                f"--checkpoint={checkpoint}",
                f"--num_envs={args.num_play_envs}",
            ]
            if args.headless:
                play_cmd.append("--headless")

            print(f"[INFO] Play command: {' '.join(play_cmd)}")
            play_proc = subprocess.run(
                play_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cf_lab_dir,
                env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )
            print(play_proc.stdout[-2000:] if len(play_proc.stdout) > 2000 else play_proc.stdout)

            # ── Step 4: Extract frames ──
            video_path = find_play_video(log_dir)
            if video_path:
                print("\n" + "=" * 60)
                print("[PHASE] Step 4: Extracting video frames")
                print("=" * 60)

                video_frames_dir = os.path.join(log_dir, "frames")
                frames_cmd = [
                    sys.executable, os.path.join(script_dir, "extract_frames.py"),
                    f"--video={video_path}",
                    f"--output-dir={video_frames_dir}",
                    f"--num-frames={args.num_frames}",
                ]
                subprocess.run(frames_cmd, cwd=cf_lab_dir)

                # Read frame manifest
                manifest_path = os.path.join(video_frames_dir, "frames_info.json")
                if os.path.isfile(manifest_path):
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    frame_paths = [fr["path"] for fr in manifest.get("frames", [])]
            else:
                print("[WARNING] No play video found.")
        else:
            print("[WARNING] No checkpoint found, skipping play.")

    # ── Step 5: Write phase report ──
    print("\n" + "=" * 60)
    print("[PHASE] Step 5: Writing phase report")
    print("=" * 60)

    # Read overrides that were applied
    overrides_applied = {}
    if args.overrides_file and os.path.isfile(args.overrides_file):
        with open(args.overrides_file) as f:
            overrides_applied = json.load(f)

    # Extract convergence summary from metrics
    convergence_summary = {}
    suspicious_patterns = []
    if metrics_data:
        for term_name, term_data in metrics_data.get("reward_terms", {}).items():
            shape = term_data.get("curve_shape", "")
            if shape in ("converged_early", "degrading", "oscillating", "still_improving"):
                convergence_summary[term_name] = {
                    "curve_shape": shape,
                    "converged_at_iteration": term_data.get("converged_at_iteration"),
                    "percent_of_training_at_convergence": term_data.get("percent_of_training_at_convergence"),
                    "final": term_data.get("final"),
                }
        # Also capture the main reward curve shape
        for key in ("Train/mean_reward", "Train/reward"):
            if key in metrics_data.get("scalars", {}):
                sdata = metrics_data["scalars"][key]
                convergence_summary["_mean_reward"] = {
                    "curve_shape": sdata.get("curve_shape"),
                    "converged_at_iteration": sdata.get("converged_at_iteration"),
                    "percent_of_training_at_convergence": sdata.get("percent_of_training_at_convergence"),
                    "final": sdata.get("final"),
                }
                break
        suspicious_patterns = metrics_data.get("suspicious_patterns", [])

    report = {
        "status": status,
        "early_stop_reason": early_stop_reason,
        "task": args.task,
        "play_task": args.play_task,
        "log_dir": log_dir,
        "iterations_completed": iterations_completed,
        "max_iterations": args.max_iterations,
        "num_envs": args.num_envs,
        "num_play_envs": args.num_play_envs,
        "overrides_applied": overrides_applied,
        "metrics": metrics_data,
        "convergence_summary": convergence_summary,
        "suspicious_patterns": suspicious_patterns,
        "video_frames_dir": video_frames_dir,
        "frame_paths": frame_paths,
        "training_time_seconds": round(training_time, 1),
        "checkpoint": find_latest_checkpoint(log_dir),
    }

    report_path = os.path.join(log_dir, "phase_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Also write to external report path for polling
    if report_path_ext:
        with open(report_path_ext, "w") as f:
            json.dump(report, f, indent=2)

    print(f"\n[DONE] Phase report: {report_path}")
    print(f"[DONE] Status: {status}")
    print(f"[DONE] Training time: {round(training_time, 1)}s")
    if early_stop_reason:
        print(f"[DONE] Early stop reason: {early_stop_reason}")

    # Print health summary for quick assessment
    if metrics_data:
        health_parts = []
        # Total reward
        for key in ("Train/mean_reward", "Train/reward"):
            if key in metrics_data.get("scalars", {}):
                s = metrics_data["scalars"][key]
                conv = s.get("converged_at_iteration")
                conv_str = f"converged@{conv}" if conv else s.get("curve_shape", "?")
                mean_r = s.get("mean_last_100")
                mean_r_str = f"{mean_r:.1f}" if isinstance(mean_r, (int, float)) else "?"
                health_parts.append(f"reward: {mean_r_str} ({conv_str})")
                break
        # Velocity tracking terms
        for term_name, term_data in metrics_data.get("reward_terms", {}).items():
            if any(kw in term_name.lower() for kw in ("track_lin_vel", "track_ang_vel")):
                val = term_data.get("mean_last_100", 0) or 0
                shape = term_data.get("curve_shape", "?")
                label = "vel_xy" if "lin" in term_name.lower() else "vel_yaw"
                status_str = "BLOCKING" if val < 0.3 else ("NEEDS_WORK" if val < 0.6 else shape)
                health_parts.append(f"{label}: {val:.3f} ({status_str})")
        # Suspicious patterns count
        health_parts.append(f"suspicious: {len(suspicious_patterns)}")
        if health_parts:
            print(f"[HEALTH] {' | '.join(health_parts)}")


def main_safe():
    """Wrapper that guarantees the external report file reflects crashes."""
    try:
        main()
    except BaseException as e:
        # If --report-path was given, write a crash report so the caller isn't stuck polling
        for arg in sys.argv:
            if arg.startswith("--report-path"):
                if "=" in arg:
                    rpath = arg.split("=", 1)[1]
                else:
                    idx = sys.argv.index(arg)
                    rpath = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
                if rpath:
                    crash_report = {
                        "status": "crashed",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    os.makedirs(os.path.dirname(os.path.abspath(rpath)) or ".", exist_ok=True)
                    with open(rpath, "w") as f:
                        json.dump(crash_report, f, indent=2)
                break
        raise


if __name__ == "__main__":
    main_safe()
