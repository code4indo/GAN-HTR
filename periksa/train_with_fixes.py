#!/usr/bin/env python3
"""
Script untuk menjalankan training GAN-HTR dengan perbaikan error
"""

import subprocess
import sys
import os

def get_safe_batch_size():
    """Tentukan batch size yang aman berdasarkan memory GPU"""
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) >= 2:
            # Dual GPU setup - bisa menggunakan batch size yang lebih besar
            return 6  # Turun dari 8 ke 6 untuk safety
        else:
            return 4  # Single GPU - batch size konservatif
    except:
        return 4  # Default fallback

def run_training():
    """Jalankan training dengan parameter yang telah diperbaiki"""
    
    print("🚀 Starting GAN-HTR training with error fixes...")
    print("=" * 60)
    
    # Parameter yang aman
    safe_batch_size = get_safe_batch_size()
    safe_epochs = 10  # Mulai dengan epochs yang lebih kecil untuk testing
    safe_lr = 0.0001
    
    print(f"📊 Safe training parameters:")
    print(f"   Batch Size: {safe_batch_size}")
    print(f"   Epochs: {safe_epochs}")
    print(f"   Learning Rate: {safe_lr}")
    print()
    
    # Konstruksi command
    cmd = [
        "poetry", "run", "python", "jnm_GAN_AHTR.py",
        "--epochs", str(safe_epochs),
        "--batch-size", str(safe_batch_size),
        "--learning-rate", str(safe_lr)
    ]
    
    print(f"🏃 Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Jalankan training
        result = subprocess.run(cmd, cwd="/home/lambda_one/tesis/GAN-HTR", check=False)
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
        else:
            print(f"\n❌ Training failed with exit code: {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error running training: {e}")
        return False
    
    return True

def check_prerequisites():
    """Periksa prerequisites sebelum training"""
    print("🔍 Checking prerequisites...")
    
    # Check if in correct directory
    if not os.path.exists("/home/lambda_one/tesis/GAN-HTR/jnm_GAN_AHTR.py"):
        print("❌ jnm_GAN_AHTR.py not found!")
        return False
    
    # Check if poetry is available
    try:
        result = subprocess.run(["poetry", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Poetry not found!")
            return False
        print(f"   ✅ Poetry found: {result.stdout.strip()}")
    except:
        print("❌ Poetry not found!")
        return False
    
    print("   ✅ All prerequisites met")
    return True

if __name__ == "__main__":
    print("🔧 GAN-HTR Training with Error Fixes")
    print("=" * 50)
    
    if not check_prerequisites():
        sys.exit(1)
    
    success = run_training()
    
    if success:
        print("\n🎉 Training script completed!")
        print("\n📝 Summary of fixes applied:")
        print("   ✅ Fixed JSON serialization error in save function")
        print("   ✅ Added iterator incarnation error recovery")
        print("   ✅ Enabled GPU memory growth to reduce register spilling")
        print("   ✅ Used safer batch size")
        print("   ✅ Added proper error handling in exception blocks")
    else:
        print("\n💡 If issues persist, try:")
        print("   1. Reduce batch size further (--batch-size 2)")
        print("   2. Use single GPU mode")
        print("   3. Check GPU memory usage")
        print("   4. Restart Python environment")
    
    sys.exit(0 if success else 1)
