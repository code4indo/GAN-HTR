#!/bin/bash

echo "🧪 Testing all dtype compatibility fixes..."

cd /home/lambda_one/tesis/GAN-HTR

# Run the dtype test
echo "Running dtype compatibility test..."
poetry run python periksa/test_dtype_fixes.py

echo ""
echo "🔍 Checking for remaining dtype issues..."

# Check for any remaining problematic patterns
if grep -n "sigmoid_cross_entropy_with_logits.*float32.*float16\|float16.*float32" jnm_GAN_AHTR.py; then
    echo "❌ Found potential dtype mismatch patterns"
else
    echo "✅ No obvious dtype mismatch patterns found"
fi

echo ""
echo "🧪 Quick syntax validation..."
if poetry run python -m py_compile jnm_GAN_AHTR.py; then
    echo "✅ Script compiles successfully!"
    echo ""
    echo "🎉 READY TO TEST!"
    echo "Run: poetry run python jnm_GAN_AHTR.py --epochs 1 --batch-size 2 --mode train"
else
    echo "❌ Compilation failed!"
    exit 1
fi
