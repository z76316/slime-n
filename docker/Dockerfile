FROM lmsysorg/sglang:dev AS base

# sgl-router
WORKDIR /sgl-workspace/sglang
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN pip install setuptools-rust wheel build
RUN . $HOME/.cargo/env && cd sgl-router/ && \
    cargo build -j 64 && python3 -m build && pip install dist/*.whl --force-reinstall

RUN pip install ray[default]
RUN pip install httpx[http2] wandb pylatexenc blobfile accelerate "mcp[cli]"
RUN pip install git+https://github.com/zhuzilin/cumem_allocator.git

# mbridge
RUN pip install git+https://github.com/ISEEKYAN/mbridge.git --no-deps

RUN TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;9.0a" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
# apex
RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git
# transformer engine
RUN pip -v install transformer_engine[pytorch]
# flash attn
# the newest version megatron supports is v2.7.4.post1
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

WORKDIR /root/
RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && \
    pip install -e .

# sandwitch norm for GLM models
COPY patch/megatron-sandwich_norm.patch /root/Megatron-LM/
RUN cd Megatron-LM && git apply megatron-sandwich_norm.patch && rm megatron-sandwich_norm.patch

# sglang patch
COPY patch/sglang.patch /sgl-workspace/sglang/
RUN cd /sgl-workspace/sglang && git apply sglang.patch && rm sglang.patch