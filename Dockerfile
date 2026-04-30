# Base image: Official PyTorch Devel (Includes full CUDA toolkit, headers, and GCC)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

RUN apt-get update && apt-get install -y \
    git ninja-build build-essential vim tmux wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --force-reinstall \
        torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
        "numpy<2.1" \
        "transformers" "wandb" "evaluate" "accelerate" \
        "ninja" "packaging<24.2"

RUN git clone https://github.com/stanford-futuredata/stk.git /workspace/_stk && \
    cd /workspace/_stk && \
    git checkout a1ddf98466730b88a2988860a9d8000fd1833301 && \
    sed -i "s/'torch>=2.3.0,<2.4'/'torch>=2.3.0'/g" setup.py && \
    pip install --no-cache-dir --no-build-isolation .

COPY ./src /workspace/src

WORKDIR /workspace/src/megablocks
RUN MAX_JOBS=4 pip install -e . --no-build-isolation --no-cache-dir --no-deps && \
    SP=$(python -c "import site; print(site.getsitepackages()[0])") && \
    find /workspace/src/megablocks -name "megablocks_ops*.so" -exec cp {} "$SP"/ \;

WORKDIR /workspace/src/OLMo
RUN pip install -e . --no-build-isolation --no-cache-dir --no-deps && \
    pip install --no-cache-dir \
        "omegaconf" "rich" "boto3" "google-cloud-storage" "tokenizers" \
        "ai2-olmo-core==0.1.0" "cached_path" "requests" "torchmetrics" \
        "smashed[remote]>=0.21.1" "safetensors" "datasets" "scikit-learn" \
        "msgspec>=0.14.0" "importlib_resources" || true

WORKDIR /workspace/src/OLMoE
RUN pip install -r requirements.txt --no-cache-dir --no-deps || true

WORKDIR /workspace
CMD ["/bin/bash"]