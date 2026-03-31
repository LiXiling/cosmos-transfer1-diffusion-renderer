# User Guide

This guide covers common workflows using [Task](https://taskfile.dev) with the Docker image.
Install Task with: `brew install go-task` or `sh -c "$(curl -ssL https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin`

## First-time setup

**1. Build the Docker image** (~15 min, downloads base image + compiles transformer-engine):

```bash
task docker:build
```

**2. Set your Hugging Face token** — needed to download the ~56 GB model weights:

```bash
export HUGGING_FACE_HUB_TOKEN=hf_your_token_here
```

Get a token at huggingface.co/settings/tokens (Read permission is enough).
The first time you run any inference task, the container will download the weights automatically into `./checkpoints/`.

---

## Relight a video

This is the main workflow. It runs three steps automatically:
1. Extract frames from your video
2. Inverse rendering → estimates lighting/material G-buffers
3. Forward rendering → applies new lighting

**Put your video inside the `asset/` folder** (required — that folder is mounted into the container):

```bash
cp /path/to/my_video.mp4 asset/input/
```

**Then run:**

```bash
task docker:relight:video VIDEO=asset/input/my_video.mp4
```

Output lands in `asset/output/`. Intermediate files are in `asset/tmp/` and can be deleted afterwards.

### Options

| Variable | Default | Description |
|---|---|---|
| `VIDEO` | *(required)* | Path to input `.mp4`, must be under `asset/` |
| `ENVLIGHT` | `0 1 2 3` | Environment map indices to use for relighting (0–3 are bundled HDRIs) |
| `NUM_FRAMES` | `57` | Number of frames to process |
| `OUTPUT_DIR` | `asset/output/` | Where to write the relit video |

**Examples:**

```bash
# Use a single lighting environment
task docker:relight:video VIDEO=asset/input/my_video.mp4 ENVLIGHT="2"

# Process more frames with custom output folder
task docker:relight:video VIDEO=asset/input/my_video.mp4 NUM_FRAMES=120 OUTPUT_DIR=asset/output/my_video/
```

### Out of memory?

On GPUs with less than 48 GB VRAM, add offload flags by opening a shell and running the inference scripts directly:

```bash
task docker:shell
# inside the container:
python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir /checkpoints \
    --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path asset/tmp/frames/my_video \
    --num_video_frames 57 --group_mode folder \
    --video_save_folder asset/tmp/gbuffers \
    --offload_diffusion_transformer --offload_tokenizer
```

---

## Other tasks

```bash
task                          # list all available tasks
task docker:shell             # interactive shell inside the container
task docker:inverse:images    # de-light a folder of images (G-buffer estimation only)
task docker:relight:images    # relight images from previously computed G-buffers
```

## Cleaning up intermediates

```bash
rm -rf asset/tmp/
```
