#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concatenate chunk videos from the forward renderer into single output videos.

The forward renderer saves one video per 57-frame chunk:
    {chunk_index}.relit_{envlight_index}.mp4

This script groups chunks by envlight index, concatenates them in order,
optionally trims to the original frame count, and removes per-chunk files.

Usage:
    python scripts/concat_relit_chunks.py \
        --output_dir asset/output/ --stem myvideo --fps 30 --total_frames 142
"""

import os
import re
import subprocess
import sys


CHUNK_PATTERN = re.compile(r"^(\d+)\.relit_(\d+)\.mp4$")


def concat_chunks(output_dir, stem, fps, total_frames=None):
    """Find chunk videos, concatenate per envlight, and clean up."""
    # Discover chunk files
    groups = {}
    for fname in os.listdir(output_dir):
        m = CHUNK_PATTERN.match(fname)
        if m:
            chunk_idx, envlight_idx = m.group(1), m.group(2)
            groups.setdefault(envlight_idx, []).append((chunk_idx, os.path.join(output_dir, fname)))

    if not groups:
        # No chunk pattern found — output may already be a single video. Nothing to do.
        return

    for envlight_idx in sorted(groups):
        chunks = sorted(groups[envlight_idx])
        files = [path for _, path in chunks]
        final_path = os.path.join(output_dir, f"{stem}.relit_{envlight_idx}.mp4")

        if len(files) == 1:
            os.rename(files[0], final_path)
            print(f"  {final_path} (single chunk)")
            continue

        # Write ffmpeg concat list
        concat_list = os.path.join(output_dir, f".concat_{envlight_idx}.txt")
        with open(concat_list, "w") as f:
            for path in files:
                f.write(f"file '{os.path.basename(path)}'\n")

        concat_basename = os.path.basename(concat_list)
        final_basename = os.path.basename(final_path)

        try:
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_basename]
            if total_frames is not None:
                cmd += ["-frames:v", str(total_frames)]
            cmd += ["-c", "copy", final_basename]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
            if result.returncode != 0:
                print(f"Warning: stream-copy concat failed, re-encoding: {result.stderr}", file=sys.stderr)
                cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_basename]
                if total_frames is not None:
                    cmd += ["-frames:v", str(total_frames)]
                cmd += ["-r", str(fps), final_basename]
                subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir, check=True)
        finally:
            if os.path.exists(concat_list):
                os.unlink(concat_list)
            for path in files:
                if os.path.exists(path):
                    os.unlink(path)

        print(f"  {final_path} ({len(files)} chunks)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Concatenate relit chunk videos into final output.")
    parser.add_argument("--output_dir", required=True, help="Directory containing chunk videos")
    parser.add_argument("--stem", required=True, help="Video stem name for the final output filename")
    parser.add_argument("--fps", type=int, required=True, help="Output video FPS")
    parser.add_argument(
        "--total_frames",
        type=int,
        default=None,
        help="Trim concatenated output to this many frames (original video length)",
    )
    args = parser.parse_args()
    concat_chunks(args.output_dir, args.stem, args.fps, args.total_frames)
