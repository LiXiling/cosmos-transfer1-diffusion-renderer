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

FROM nvcr.io/nvidia/pytorch:25.03-py3

# Install system tools + CUDA devel headers required by transformer-engine and apex
RUN apt-get update && apt-get install -y --no-install-recommends \
        git tree ffmpeg wget \
        cuda-nvcc-12-8 \
        cuda-cudart-dev-12-8 \
        libcublas-dev-12-8 \
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

# 3. transformer-engine (requires torch + CUDA headers at build time)
RUN uv pip install --system --no-cache --no-build-isolation "transformer-engine[pytorch]==1.12.0"

# 4. apex (compiled CUDA extensions — uses pip directly; uv doesn't support --build-option)
# Set arch list explicitly so apex cross-compiles for Blackwell (10.0 = RTX 5090) and older GPUs
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0"
RUN git clone --depth 1 https://github.com/NVIDIA/apex /tmp/apex \
    && pip install --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" \
        /tmp/apex \
    && rm -rf /tmp/apex

# Copy project code
COPY . /workspace

ENV PYTHONPATH=/workspace

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
