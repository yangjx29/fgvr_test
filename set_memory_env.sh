#!/bin/bash
# 显存优化环境变量配置脚本
# 使用方法: source set_memory_env.sh

# PyTorch CUDA内存分配器优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# 减少碎片化
export CUDA_LAUNCH_BLOCKING=0

# 限制OMP线程数，减少CPU内存占用
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "✓ 已设置显存优化环境变量:"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"

