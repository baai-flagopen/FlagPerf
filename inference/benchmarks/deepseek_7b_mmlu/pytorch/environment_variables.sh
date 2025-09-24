#!/bin/bash
# Environment variables for deepseek_7b_mmlu benchmark

# Set CUDA device if needed
# export CUDA_VISIBLE_DEVICES=0

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Set transformers cache directory if needed
# export TRANSFORMERS_CACHE=/tmp/transformers_cache

# Other environment variables can be added here
echo "Environment variables loaded for deepseek_7b_mmlu"
