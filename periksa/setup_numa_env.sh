#!/bin/bash
# Script untuk mengatur environment variables untuk mengatasi NUMA warnings
# File: setup_numa_env.sh

echo "🔧 Setting up environment for NUMA warning suppression..."

# Export environment variables
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONIOENCODING=utf-8
export TF_DISABLE_LAYOUT_OPTIMIZER=1

echo "✅ Environment variables set:"
echo "   - TF_CPP_MIN_LOG_LEVEL = $TF_CPP_MIN_LOG_LEVEL"
echo "   - TF_FORCE_GPU_ALLOW_GROWTH = $TF_FORCE_GPU_ALLOW_GROWTH"
echo "   - TF_GPU_ALLOCATOR = $TF_GPU_ALLOCATOR"
echo "   - CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

echo ""
echo "🚀 Ready to run training without NUMA warnings!"
echo "💡 Run your training script now:"
echo "   poetry run python jnm_GAN_AHTR.py"
echo ""
echo "🔍 To verify GPU setup:"
echo "   poetry run python periksa/test_numa_fix.py"
