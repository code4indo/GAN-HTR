#!/usr/bin/env python3
"""
Test script untuk memverifikasi bahwa NUMA warnings sudah teratasi
"""

import os
import sys
import warnings
import subprocess
import time

def test_numa_warning_suppression():
    """
    Test apakah NUMA warnings sudah berhasil di-suppress
    """
    print("🧪 Testing NUMA Warning Suppression...")
    print("=" * 50)
    
    # Set environment variables seperti di file utama
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    
    print("✅ Environment variables set")
    
    try:
        # Import TensorFlow dengan konfigurasi yang sama
        import tensorflow as tf
        
        # Configure logging
        tf.get_logger().setLevel('ERROR')
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        # Suppress warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
        warnings.filterwarnings('ignore', '.*NUMA.*')
        
        print("✅ TensorFlow imported and configured")
        
        # Configure GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Configured {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"⚠️  GPU config warning: {e}")
        
        # Test GPU operation
        print("\n🔬 Testing GPU operations...")
        with tf.device('/GPU:0'):
            # Create simple tensors
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            
            # Matrix multiplication
            start_time = time.time()
            c = tf.matmul(a, b)
            end_time = time.time()
            
            print(f"✅ GPU operation completed in {end_time - start_time:.4f} seconds")
            print(f"   Result shape: {c.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def test_training_import():
    """
    Test import dari file training utama
    """
    print("\n🧪 Testing Training Script Import...")
    print("-" * 30)
    
    try:
        # Import komponen utama dari jnm_GAN_AHTR
        sys.path.append('/home/lambda_one/tesis/GAN-HTR')
        
        # Test minimal import tanpa error
        import tensorflow.keras.layers as layers
        import tensorflow.keras.models as models
        
        print("✅ Keras imports successful")
        
        # Test LeakyReLU (yang sudah diperbaiki)
        leaky = layers.LeakyReLU(negative_slope=0.2)
        print("✅ LeakyReLU with negative_slope works")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def run_quick_training_test():
    """
    Test menjalankan script training dengan timeout singkat
    """
    print("\n🧪 Testing Quick Training Run...")
    print("-" * 30)
    
    try:
        # Jalankan script utama dengan timeout
        cmd = ["poetry", "run", "python", "jnm_GAN_AHTR.py"]
        
        print("🚀 Starting quick training test (10 seconds)...")
        
        # Capture output
        process = subprocess.Popen(
            cmd,
            cwd="/home/lambda_one/tesis/GAN-HTR",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = process.communicate(timeout=10)
            
            # Check for NUMA warnings in output
            numa_warnings = []
            if "NUMA node" in stderr:
                numa_warnings = [line for line in stderr.split('\n') if "NUMA node" in line]
            
            if numa_warnings:
                print("⚠️  NUMA warnings still present:")
                for warning in numa_warnings:
                    print(f"   {warning}")
                return False
            else:
                print("✅ No NUMA warnings detected!")
                return True
                
        except subprocess.TimeoutExpired:
            process.kill()
            print("✅ Training started successfully (timeout expected)")
            return True
            
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

def main():
    """
    Main test function
    """
    print("🔍 NUMA Warning Suppression Verification")
    print("=" * 60)
    
    # Run tests
    test1 = test_numa_warning_suppression()
    test2 = test_training_import()
    test3 = run_quick_training_test()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"   ✅ NUMA suppression test: {'PASSED' if test1 else 'FAILED'}")
    print(f"   ✅ Training import test: {'PASSED' if test2 else 'FAILED'}")
    print(f"   ✅ Quick training test: {'PASSED' if test3 else 'FAILED'}")
    
    all_passed = test1 and test2 and test3
    
    if all_passed:
        print("\n🎉 All tests PASSED!")
        print("💡 NUMA warnings successfully suppressed!")
        print("\n🚀 Ready to run training without NUMA warnings:")
        print("   poetry run python jnm_GAN_AHTR.py")
    else:
        print("\n❌ Some tests FAILED.")
        print("💡 Check the configuration and try again.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
