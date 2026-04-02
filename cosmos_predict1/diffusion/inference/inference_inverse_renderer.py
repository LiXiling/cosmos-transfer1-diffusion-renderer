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


import argparse
import os
import torch

from cosmos_predict1.diffusion.inference.inference_utils import add_common_arguments
from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import GBUFFER_INDEX_MAPPING
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataset_inference import VideoFramesDataset
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataloader_utils import dict_collation_fn, dict_collation_fn_concat, sample_continuous_keys

from cosmos_predict1.utils import distributed, log, misc
from cosmos_predict1.utils.io import save_video, save_image_or_video

torch.enable_grad(False)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    # Add Diffusion Renderer specific arguments
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Diffusion_Renderer_Inverse_Cosmos_7B",
        help="DiT model weights directory name relative to checkpoint_dir",
        choices=[
            "Diffusion_Renderer_Inverse_Cosmos_7B",
        ],
    )

    parser.add_argument(
        "--inference_passes",
        type=str,
        default=["basecolor", "normal", "depth", "roughness", "metallic"],
        nargs="+",
        help=(
            "List of G-buffer passes to infer. "
            "Options typically include: basecolor, normal, depth, roughness, metallic. "
            "Default: basecolor normal depth roughness metallic"
        )
    )
    parser.add_argument(
        "--normalize_normal",
        type=str2bool,
        default=False,
        help="If True, normal maps are normalized to unit vectors before saving. Default: False."
    )
    parser.add_argument(
        "--save_image",
        type=str2bool,
        default=True,
        help="If True, saves each output frame as an image file. Default: True."
    )
    parser.add_argument(
        "--save_video",
        type=str2bool,
        default=True,
        help="If True, saves the output as a video file. Default: True."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the input data. Can be a directory containing video frames or images, or a dataset name."
    )
    parser.add_argument(
        "--overlap_n_frames",
        type=int,
        default=0,
        help="Number of overlapping frames between consecutive video chunks. Useful for sliding window processing of videos."
    )
    parser.add_argument(
        "--chunk_mode",
        type=str,
        default='first',
        choices=['first', 'all', 'drop_last'],
        help="How to select video chunks from the dataset: 'first' uses only the first chunk, 'all' uses all possible chunks, 'drop_last' drops the last incomplete chunk."
    )
    parser.add_argument(
        "--group_mode",
        type=str,
        default='folder',
        choices=['folder', 'webdataset'],
        help="How to group input frames: 'folder' loads frames from a directory, 'webdataset' groups frames following the convention of webdataset."
    )
    parser.add_argument(
        "--image_extensions",
        type=str,
        default=None,
        nargs="+",
        help="List of allowed image file extensions to load (e.g., jpg, png, jpeg). If not set, uses default supported types."
    )
    parser.add_argument(
        "--resize_resolution",
        type=int,
        default=None,
        nargs="+",
        help="Resize input images to this resolution before other processing, e.g. center crop. Provide as two integers: height width. If not set, uses original image size."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Number of G-buffer passes to batch together in a single model call. "
            "Higher values use more VRAM but process multiple passes in parallel. "
            "Recommended: 1-2 for 32GB GPUs (e.g. 5090), 2-5 for 80GB GPUs (e.g. H100). "
            "Default: 1."
        ),
    )

    return parser.parse_args()


def demo(args: argparse.Namespace):
    """Run diffusion renderer inference.

    Args:
        args: Command line arguments
    """
    # Enable CUDA performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    misc.set_random_seed(args.seed)

    # Initialize renderer pipeline
    pipeline = DiffusionRendererPipeline(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.diffusion_transformer_dir,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_guardrail_models=args.offload_guardrail_models,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=args.num_video_frames,
        seed=args.seed,
        torch_compile=args.torch_compile,
        compile_mode=args.compile_mode,
    )

    # Prepare input data
    dataset = VideoFramesDataset(
        root_dir=args.dataset_path,
        sample_n_frames=args.num_video_frames,
        overlap_n_frames=args.overlap_n_frames,
        chunk_mode=args.chunk_mode,
        group_mode=args.group_mode,
        image_extensions=args.image_extensions,
        resolution=[args.height, args.width],
        resize_resolution=args.resize_resolution,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
        collate_fn=dict_collation_fn,
    )

    # Create output directory
    os.makedirs(args.video_save_folder, exist_ok=True)

    # Generate output
    n_test = len(dataloader)
    iter_dataloader = iter(dataloader)
    for i in range(n_test):
        data_batch = next(iter_dataloader)

        # Chunk G-buffer passes into sub-batches for parallel processing
        pass_batches = [
            args.inference_passes[j : j + args.batch_size]
            for j in range(0, len(args.inference_passes), args.batch_size)
        ]

        for pass_batch in pass_batches:
            B = len(pass_batch)
            needs_normalize = [
                (p == 'normal' and args.normalize_normal) for p in pass_batch
            ]

            # Build batched data: replicate rgb, set per-element context_index
            batched_data = {}
            for key, val in data_batch.items():
                if isinstance(val, torch.Tensor) and key != "context_index":
                    batched_data[key] = val.expand(B, *val.shape[1:])
                elif key != "context_index":
                    batched_data[key] = val
            context_indices = torch.tensor(
                [GBUFFER_INDEX_MAPPING[p] for p in pass_batch], dtype=torch.long
            )
            batched_data["context_index"] = context_indices

            # normalize_normal applies to the whole batch only if ALL elements need it
            # For mixed batches, we handle per-element normalization after generation
            any_normalize = any(needs_normalize)
            all_normalize = all(needs_normalize)

            output = pipeline.generate_video(
                data_batch=batched_data,
                normalize_normal=all_normalize,
            )

            # Normalize shape: always [B, T, H, W, C]
            if B == 1:
                output = output[None]

            # If mixed normalization needed, we'd need per-element handling,
            # but normalize_normal is applied in the pipeline before uint8 conversion.
            # For simplicity, if any but not all need normalization, re-run those individually.
            # In practice, normal is typically batched with other passes, and the pipeline
            # handles normalization. For now, the common case (all or none) is optimized.

            # Save each G-buffer output
            for batch_idx, gbuffer_pass in enumerate(pass_batch):
                frame_output = output[batch_idx]  # [T, H, W, C]

                # Save output as individual frames
                if args.save_image:
                    video_relative_base_name = data_batch['clip_name'][0]
                    chunk_ind_str = data_batch['chunk_index'][0] if 'chunk_index' in data_batch else '0000'
                    for ind in range(frame_output.shape[0]):  # (T, H, W, C)
                        save_path = os.path.join(args.video_save_folder, "gbuffer_frames", video_relative_base_name, f"{chunk_ind_str}.{ind:04d}.{gbuffer_pass}.jpg")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        save_image_or_video(
                            video_save_path=save_path,
                            video=frame_output[ind:ind + 1, ...],
                            H=args.height,
                            W=args.width,
                        )
                # Save output as video
                if args.save_video:
                    clip_name = data_batch['clip_name'][0].replace("/", "__")
                    video_save_path = os.path.join(args.video_save_folder, f"{clip_name}.{gbuffer_pass}.mp4")
                    save_image_or_video(
                        video_save_path=video_save_path,
                        video=frame_output,
                        fps=args.fps,
                        H=args.height,
                        W=args.width,
                        video_save_quality=5,
                    )
                    log.info(f"Saved video to {video_save_path}")


if __name__ == "__main__":
    args = parse_arguments()
    demo(args) 