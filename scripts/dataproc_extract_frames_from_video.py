# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import subprocess
import sys


def extract_frames_from_folder(
    input_folder, output_base_dir, frame_rate=10, resize=None, max_frames=None
):
    """
    Extract frames from all MP4 videos in a folder and save them as images.

    :param input_folder: Path to the folder containing MP4 video files.
    :param output_base_dir: Directory where the frames will be saved.
    :param frame_rate: Number of frames to extract per second.
    """
    # Find all MP4 files in the input folder
    video_files = glob.glob(os.path.join(input_folder, "*.mp4")) + glob.glob(
        os.path.join(input_folder, "*.MP4")
    )

    if not video_files:
        print(f"No MP4 files found in {input_folder}")
        return

    for video_file in video_files:
        # Get the video filename without extension
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        output_dir = os.path.join(output_base_dir, video_name)

        # Extract frames for the current video
        extract_frames(video_file, output_dir, frame_rate, resize, max_frames)


def extract_frames(
    video_path, output_dir, frame_rate=None, resize=None, max_frames=None
):
    """
    Extract frames from a video and save them as images using ffmpeg.
    Supports all codecs ffmpeg can decode (including AV1, HEVC, etc.).

    :param video_path: Path to the input video file.
    :param output_dir: Directory where the frames will be saved.
    :param frame_rate: Number of frames to extract per second.
    :param resize: Optional tuple (width, height) to resize frames.
    :param max_frames: Maximum number of frames to extract.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build ffmpeg filter chain
    vf_parts = []
    if frame_rate is not None:
        vf_parts.append(f"fps={frame_rate}")
    if resize is not None:
        vf_parts.append(f"scale={resize[0]}:{resize[1]}")

    cmd = ["ffmpeg", "-y", "-i", video_path]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-q:v", "2", os.path.join(output_dir, "frame_%05d.jpg")]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: Could not extract frames from {video_path}")
        print(result.stderr)
        sys.exit(1)

    # Count saved frames
    saved = len(glob.glob(os.path.join(output_dir, "frame_*.jpg")))
    print(f"Frames extracted from {video_path}. Total frames saved: {saved}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the folder containing MP4 video files.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Directory where the frames will be saved.",
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=None,
        help="Number of frames to extract per second.",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Resize extracted frames to WIDTHxHEIGHT (e.g., 1280x704).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract per video (optional).",
    )

    args = parser.parse_args()

    # Parse resize argument if provided
    resize = None
    if args.resize:
        try:
            width, height = map(int, args.resize.lower().split("x"))
            resize = (width, height)
        except ValueError:
            raise ValueError(
                "Invalid format for --resize. Use WIDTHxHEIGHT, e.g., 640x480."
            )

    # Call the main function with parsed arguments
    extract_frames_from_folder(
        args.input_folder, args.output_folder, args.frame_rate, resize, args.max_frames
    )
