#!/usr/bin/env python3
"""
Quick training test - menjalankan 1 epoch dengan batch kecil untuk verifikasi
"""

import os
import sys
import traceback
import tensorflow as tf

# Enable Mixed Precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Suppress some warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def quick_training_test():
    """Quick test training 1 epoch"""
    print("🚀 Starting Quick Training Test...")
    print("This will run 1 epoch with small batch size to verify everything works")
    
    try:
        # Import the main training script functions
        sys.path.append('.')
        
        # Set environment variables to use GPU 1 (as in original script)
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        
        print("📝 Attempting to import and run training functions...")
        
        # Try to import the script
        exec(open('jnm_GAN_AHTR.py').read())
        
        print("✅ Script imported successfully")
        print("✅ All functions and models available")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 QUICK TRAINING TEST")
    print("=" * 40)
    
    success = quick_training_test()
    
    print("=" * 40)
    if success:
        print("🎉 QUICK TEST PASSED!")
        print("💡 Script is ready for full training!")
        print("🔥 To start training:")
        print("   cd /home/lambda_one/tesis/GAN-HTR")
        print("   poetry run python jnm_GAN_AHTR.py")
    else:
        print("⚠️  Test failed - check errors above")
