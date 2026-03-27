# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extract evenly-spaced frames from an MP4 video as PNG images.

Usage:
    python .claude/skills/auto_train/resources/extract_frames.py --video <path.mp4> --output-dir <dir> [--num-frames 8]
"""

import argparse
import json
import os
import sys

import cv2


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video as PNG images.")
    parser.add_argument("--video", type=str, required=True, help="Path to MP4 video file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save extracted frames.")
    parser.add_argument("--num-frames", type=int, default=12, help="Number of frames to extract (default: 12).")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        print(f"[ERROR] Video has no frames: {video_path}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    # Skip early frames (Isaac Sim renders black for the first few frames)
    skip_start = max(int(total_frames * 0.1), 2)
    usable_frames = total_frames - skip_start
    num_frames = min(args.num_frames, usable_frames)

    if usable_frames <= 0:
        print(f"[ERROR] Video too short to extract frames after skipping initial black frames.", file=sys.stderr)
        cap.release()
        sys.exit(1)

    # Calculate evenly-spaced frame indices from the usable range
    if num_frames == 1:
        indices = [skip_start + usable_frames // 2]
    else:
        indices = [skip_start + int(i * (usable_frames - 1) / (num_frames - 1)) for i in range(num_frames)]

    frames_info = []
    extracted = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Could not read frame {idx}", file=sys.stderr)
            continue

        extracted += 1
        filename = f"frame_{extracted:03d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)

        timestamp_sec = idx / fps if fps > 0 else 0.0
        frames_info.append({
            "path": filepath,
            "filename": filename,
            "frame_index": idx,
            "timestamp_sec": round(timestamp_sec, 3),
        })

    cap.release()

    # Write manifest
    manifest = {
        "video_path": video_path,
        "total_video_frames": total_frames,
        "fps": fps,
        "num_frames_extracted": extracted,
        "frames": frames_info,
    }

    manifest_path = os.path.join(output_dir, "frames_info.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Extracted {extracted} frames to: {output_dir}")
    print(f"[INFO] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
