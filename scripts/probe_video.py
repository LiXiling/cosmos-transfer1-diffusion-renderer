#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe video metadata for the rendering pipeline.

Reports video properties adjusted to model-compatible values:
  fps          – native frame rate (rounded to nearest integer)
  width        – native width rounded down to nearest multiple of --divisor
  height       – native height rounded down to nearest multiple of --divisor
  total_frames – total number of frames in the video

Usage:
  python scripts/probe_video.py video.mp4              # all fields
  python scripts/probe_video.py video.mp4 fps          # single field
  python scripts/probe_video.py video.mp4 width --divisor 16
"""

import json
import subprocess
import sys


def probe(video_path, divisor=16):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"ffprobe error: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    stream = data["streams"][0]

    # FPS — r_frame_rate is a fraction like "30000/1001" or "24/1"
    num, den = map(int, stream["r_frame_rate"].split("/"))
    fps = round(num / den)

    # Resolution — round down to nearest multiple of divisor
    width = (int(stream["width"]) // divisor) * divisor
    height = (int(stream["height"]) // divisor) * divisor

    # Total frames
    nb_frames = stream.get("nb_frames", "N/A")
    if nb_frames not in (None, "N/A", ""):
        total_frames = int(nb_frames)
    else:
        duration = float(data.get("format", {}).get("duration", 0))
        total_frames = round(duration * fps)

    return {"fps": fps, "width": width, "height": height, "total_frames": total_frames}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Probe video metadata for the rendering pipeline.")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("field", nargs="?", help="Single field to output (fps, width, height, total_frames)")
    parser.add_argument(
        "--divisor",
        type=int,
        default=16,
        help="Round dimensions down to nearest multiple of this value (default: 16). "
        "The Cosmos diffusion renderer requires dimensions divisible by 16 "
        "(8x spatial compression * 2x patch size).",
    )
    args = parser.parse_args()

    info = probe(args.video, args.divisor)

    if args.field:
        if args.field not in info:
            sys.exit(f"Unknown field '{args.field}'. Choose from: {', '.join(info)}")
        print(info[args.field])
    else:
        for k, v in info.items():
            print(f"{k}={v}")
