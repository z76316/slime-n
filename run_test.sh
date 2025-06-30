# set -euxo pipefail

# # hf checkpoint
# huggingface-cli download Qwen/Qwen3-4B --local-dir ../model/Qwen3-4B

# # train data
# huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
#   --local-dir ../data/dapo-math-17k

# # eval data
# huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
#   --local-dir ../data/aime-2024



########convert hf to torch dist########

# # PYTHONPATH=/home/yushensu/projects/slime/Megatron-LM python tools/convert_hf_to_torch_dist.py \
# #     --hf-checkpoint ../model/Qwen3-4B \
# #     --save ../model/Qwen3-4B_torch_dist




# export LLVM_SYMBOLIZER_PATH=/opt/rocm/llvm/bin/llvm-symbolizer
# export PATH=$PATH:/opt/rocm/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib/
# export GPU_ARCHS="gfx942"
# export PYTORCH_ROCM_ARCH="gfx90a;gfx942"

# # AMD ROCm setting
# # export HSA_ENABLE_SDMA=0
# export AMD_SERIALIZE_KERNEL=3
# export TORCH_USE_HIP_DSA=1 
# export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  
# export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7    

# # Enable the features but loss some efficiency

# # Need to add:
# # --no-gradient-accumulation-fusion
# # to the below:
# # GPT_MODEL_ARGS=(
# #     --normalization RMSNorm
# #     --swiglu
# #     --disable-bias-linear
# #     --seq-length 1
# #     --max-position-embeddings 40960
# #     --attention-backend auto # Can use (flash/fused/unfused/local)
# #     --position-embedding-type rope
# #     --kv-channels 128
# #     --qk-layernorm
# #     --group-query-attention
# # )

# cd /home/yushensu/projects/slime/Pai-Megatron-Patch-amd_version/toolkits/distributed_checkpoints_convertor
# bash scripts/qwen3/run_8xH20.sh \
# 4B \
# /home/yushensu/projects/model/Qwen3-4B \
# /home/yushensu/projects/model/Qwen3-4B_torch_dist \
# false \
# true \
# bf16 

###################################









# # git checkout f612bdf
# git checkout 83f0eba

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HIP_VISIBLE_DEVICES=0,1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
# export AMD_SERIALIZE_KERNEL=3 # for rocm debug
# export AMD_SERIALIZE_KERNEL=1

# Run Training
cd /home/yushensu/projects/slime
bash scripts/run-qwen3-4B.sh

