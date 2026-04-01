# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cosmos-Transfer1-DiffusionRenderer is an NVIDIA video relighting framework built on Cosmos World Foundation Models. It implements a two-stage pipeline: **inverse rendering** (RGB → G-buffers: albedo, metallic, roughness, normal, depth) and **forward rendering** (G-buffers + environment maps → relit RGB).

## Setup

```bash
conda env create --file cosmos-predict1.yaml
conda activate cosmos-predict1
pip install -r requirements.txt
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
pip install transformer-engine[pytorch]
```

Verify environment: `CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py`

Download weights (~56GB): `CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_renderer_checkpoints.py --checkpoint_dir checkpoints`

## Code Formatting

```bash
black --line-length 120 --target-version py310 .   # excludes third_party/
isort --profile black --line-length 120 .
```

## Inference Commands

All inference commands require `CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd)` prefix. Add `--offload_diffusion_transformer --offload_tokenizer` for low VRAM (<48GB).

**Inverse rendering (images):**
```bash
python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path=asset/examples/image_examples/ --num_video_frames 1 --group_mode webdataset \
    --video_save_folder=asset/example_results/image_delighting/ --save_video=False
```

**Forward rendering (relighting):**
```bash
python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/example_results/image_delighting/gbuffer_frames --num_video_frames 1 \
    --envlight_ind 0 1 2 3 --use_custom_envmap=True \
    --video_save_folder=asset/example_results/image_relighting/
```

**Video workflow:** Extract frames first with `scripts/dataproc_extract_frames_from_video.py`, then run inverse/forward rendering with `--num_video_frames 57 --group_mode folder`.

## Architecture

### Two-Stage Rendering Pipeline

The pipeline flows through two separate diffusion transformer models:

1. **Inverse Renderer** — [`inference_inverse_renderer.py`](cosmos_predict1/diffusion/inference/inference_inverse_renderer.py): Takes RGB frames, outputs G-buffer maps (albedo/basecolor, normal, depth, roughness, metallic). Controlled by `--inference_passes` to select which G-buffers to compute.

2. **Forward Renderer** — [`inference_forward_renderer.py`](cosmos_predict1/diffusion/inference/inference_forward_renderer.py): Takes G-buffer frames + environment maps (HDRIs), outputs relit RGB video. Supports `--rotate_light` for rotating environment animations and `--use_custom_envmap=False` for random illumination.

### Core Components

- **`DiffusionRendererPipeline`** ([`diffusion_renderer_pipeline.py`](cosmos_predict1/diffusion/inference/diffusion_renderer_pipeline.py)): Main orchestrator wrapping the base `WorldGenerationPipeline`.

- **`WorldGenerationPipeline`** ([`world_generation_pipeline.py`](cosmos_predict1/diffusion/inference/world_generation_pipeline.py), ~1970 lines): Base generation pipeline shared across all model variants (text2world, video2world, multiview, etc.). Handles tokenization, denoising, and decoding.

- **`DiffusionRendererModel`** ([`model/model_diffusion_renderer.py`](cosmos_predict1/diffusion/model/model_diffusion_renderer.py)): Wraps the DiT architecture for both inverse and forward rendering modes.

- **`GeneralDiTDiffusionRenderer`** ([`networks/general_dit_diffusion_renderer.py`](cosmos_predict1/diffusion/networks/general_dit_diffusion_renderer.py)): Diffusion Transformer network specialized for rendering, extending the base `GeneralDiT` ([`networks/general_dit.py`](cosmos_predict1/diffusion/networks/general_dit.py)).

- **`VideoDiffusionRendererCondition`** ([`conditioner.py`](cosmos_predict1/diffusion/conditioner.py)): Conditioning mechanism that encodes G-buffer maps or RGB frames as spatial conditions for the DiT.

### Data Loading

[`diffusion_renderer_utils/dataset_inference.py`](cosmos_predict1/diffusion/inference/diffusion_renderer_utils/dataset_inference.py) provides `VideoFramesDataset` and `VideoGBufferDataset` for loading inference data. `--group_mode webdataset` (single images) vs `--group_mode folder` (video frame sequences) controls how input data is grouped.

### Configuration System

Configs are registered via a registry pattern in [`config/registry.py`](cosmos_predict1/diffusion/config/registry.py). Renderer-specific configs live in [`config/diffusion_renderer_config.py`](cosmos_predict1/diffusion/config/diffusion_renderer_config.py).

### Utility Modules

- [`diffusion_renderer_utils/rendering_utils.py`](cosmos_predict1/diffusion/inference/diffusion_renderer_utils/rendering_utils.py): G-buffer operations, cubemap/latlong conversions, vector math
- [`inference_utils.py`](cosmos_predict1/diffusion/inference/inference_utils.py): Model loading, video processing, argument parsing shared across inference scripts
- [`module/pretrained_vae.py`](cosmos_predict1/diffusion/module/pretrained_vae.py): VAE encoder/decoder for latent space operations
