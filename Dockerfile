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

FROM nvcr.io/nvidia/pytorch:25.06-py3

# Install system tools + CUDA devel headers (for nvdiffrast compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git tree ffmpeg wget \
        cuda-nvcc-12-9 \
        cuda-cudart-dev-12-9 \
        libcublas-dev-12-9 \
    && rm -rf /var/lib/apt/lists/*

# Ensure libcuda.so (unversioned) is available for build-time linking
RUN ln -sf /lib64/libcuda.so.1 /lib64/libcuda.so 2>/dev/null || true

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# CUDA build environment (used for transformer-engine and apex compilation)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# Allow uv to install into the PEP-668 externally-managed system Python
ENV UV_BREAK_SYSTEM_PACKAGES=1

WORKDIR /workspace

# --- Dependency installation (cached as separate layers) ---

# NOTE: torch + torchvision are already installed in the NGC base image at the
# correct CUDA version; no need to reinstall them.

# 2. All other project dependencies (via pyproject.toml)
# Stub package dir is required so setuptools can resolve the package during dep install;
# the real source is overlaid by COPY . /workspace later.
COPY pyproject.toml .
RUN mkdir -p cosmos_predict1 && \
    uv pip install --system --no-cache .

# 3. transformer-engine — already pre-installed in the NGC base image at the
#    correct version; no need to reinstall.

# 4. nvdiffrast (CUDA extension — build from source)
RUN git clone --depth 1 https://github.com/NVlabs/nvdiffrast /tmp/nvdiffrast \
    && TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0" \
       pip install --no-cache-dir --no-build-isolation /tmp/nvdiffrast \
    && rm -rf /tmp/nvdiffrast

# NOTE: apex is pre-installed in the NGC PyTorch base image; no need to build from source.
# It is only used for FusedAdam (training) and is guarded by try/except, so inference
# works even without it.

# Copy project code
COPY . /workspace

ENV PYTHONPATH=/workspace

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
