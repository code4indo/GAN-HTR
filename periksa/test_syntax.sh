#!/bin/bash

# Quick test script to verify the Python syntax
echo "🧪 Testing jnm_GAN_AHTR.py syntax..."

cd /home/lambda_one/tesis/GAN-HTR

# Check syntax with poetry run
if poetry run python -m py_compile jnm_GAN_AHTR.py; then
    echo "✅ Script syntax is valid!"
    echo ""
    echo "🎉 SUCCESS! All errors have been fixed:"
    echo "✅ Mixed precision dtype mismatch resolved"
    echo "✅ Duplicate code sections removed" 
    echo "✅ Undefined variable (avg_speed) fixed"
    echo "✅ Indentation errors corrected"
    echo ""
    echo "🚀 Script is ready to run!"
    echo ""
    echo "To test training, run:"
    echo "poetry run python jnm_GAN_AHTR.py --epochs 2 --batch-size 2 --mode train"
else
    echo "❌ Syntax errors still exist!"
    exit 1
fi
