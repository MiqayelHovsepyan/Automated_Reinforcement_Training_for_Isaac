# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Block until a training phase completes, then print the final report.

This script replaces the sleep-poll loop that wastes context window cycles.
Claude calls this once with a long Bash timeout instead of 50-80 poll cycles.

Usage:
    python .claude/skills/auto_train/resources/wait_for_phase.py \
        --report-path=.claude/skills/auto_train/experiments/.scratch/current_phase_report.json \
        --poll-interval=30 --timeout=7200
"""

import argparse
import json
import os
import sys
import time


_TERMINAL_STATUSES = {"completed", "early_stopped", "failed", "crashed", "validation_error"}


def main():
    parser = argparse.ArgumentParser(description="Block until a training phase completes.")
    parser.add_argument("--report-path", type=str, required=True, help="Path to the phase report JSON file.")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between checks (default: 30).")
    parser.add_argument("--timeout", type=int, default=7200, help="Max seconds to wait (default: 7200 = 2 hours).")
    args = parser.parse_args()

    report_path = os.path.abspath(args.report_path)
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed > args.timeout:
            print(json.dumps({"status": "wait_timeout", "elapsed_seconds": round(elapsed)}), flush=True)
            sys.exit(1)

        if os.path.isfile(report_path):
            try:
                with open(report_path) as f:
                    report = json.load(f)
                status = report.get("status", "unknown")

                if status in _TERMINAL_STATUSES:
                    # Training is done — print the full report and exit
                    print(json.dumps(report, indent=2), flush=True)
                    return

                # Still running — print a one-line progress summary to stderr
                progress = report.get("progress", {})
                cur = progress.get("current_iteration", "?")
                mx = progress.get("max_iterations", "?")
                pct = progress.get("percent_complete", "?")
                reward = progress.get("mean_reward", "?")
                eta = progress.get("eta_seconds")
                eta_str = f"{eta}s" if eta is not None else "?"
                print(
                    f"[WAIT] {pct}% ({cur}/{mx}) | reward: {reward} | ETA: {eta_str} | elapsed: {round(elapsed)}s",
                    file=sys.stderr,
                    flush=True,
                )
            except (json.JSONDecodeError, OSError):
                pass  # File being written, retry next cycle
        else:
            print(f"[WAIT] Report file not found yet, waiting... ({round(elapsed)}s elapsed)", file=sys.stderr, flush=True)

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
