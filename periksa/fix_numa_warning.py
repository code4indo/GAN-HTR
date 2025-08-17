#!/usr/bin/env python3
"""
Solusi untuk mengatasi NUMA node warning pada CUDA/TensorFlow.
Warning: successful NUMA node read from SysFS had negative value (-1)

Masalah ini umum terjadi pada:
1. Sistem dengan GPU NVIDIA
2. Docker containers
3. Virtual machines
4. Sistem dengan konfigurasi NUMA yang tidak standar
"""

import os
import subprocess
import sys

def set_cuda_environment_variables():
    """
    Set environment variables untuk mengatasi NUMA warnings
    """
    print("🔧 Setting CUDA environment variables...")
    
    # Environment variables untuk mengatasi NUMA warnings
    cuda_env_vars = {
        'TF_CPP_MIN_LOG_LEVEL': '2',  # Suppress INFO dan WARNING logs
        'CUDA_VISIBLE_DEVICES': '0,1',  # Sesuaikan dengan GPU yang tersedia
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_GPU_ALLOCATOR': 'cuda_malloc_async'
    }
    
    for key, value in cuda_env_vars.items():
        os.environ[key] = value
        print(f"   ✅ {key} = {value}")
    
    print("✅ CUDA environment variables set!")

def check_gpu_info():
    """
    Check informasi GPU yang tersedia
    """
    print("\n🔍 Checking GPU information...")
    
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')
            print("📊 GPU Information:")
            for i, info in enumerate(gpu_info):
                print(f"   GPU {i}: {info}")
        else:
            print("⚠️  nvidia-smi not available or failed")
            
    except Exception as e:
        print(f"⚠️  Could not get GPU info: {e}")

def create_tf_config_suppressor():
    """
    Buat konfigurasi TensorFlow untuk suppress warnings
    """
    print("\n🔧 Creating TensorFlow configuration...")
    
    tf_config = '''
import os
import warnings
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Suppress specific NUMA warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Configured {len(gpus)} GPU(s) with memory growth")
    except RuntimeError as e:
        print(f"⚠️  GPU configuration error: {e}")

# Suppress NUMA warnings specifically
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
'''
    
    return tf_config

def test_tensorflow_gpu():
    """
    Test TensorFlow GPU setup setelah konfigurasi
    """
    print("\n🧪 Testing TensorFlow GPU setup...")
    
    try:
        import tensorflow as tf
        
        # Suppress logs untuk test ini
        tf.get_logger().setLevel('ERROR')
        
        # Check GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        print(f"📊 TensorFlow detects {len(gpus)} GPU(s)")
        
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
        
        # Simple operation test
        if gpus:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"✅ GPU operation test successful: {c.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow GPU test failed: {e}")
        return False

def main():
    """
    Main function untuk mengatasi NUMA warnings
    """
    print("🚀 NUMA Node Warning Solution")
    print("=" * 50)
    
    # Step 1: Set environment variables
    set_cuda_environment_variables()
    
    # Step 2: Check GPU info
    check_gpu_info()
    
    # Step 3: Create TF config
    tf_config = create_tf_config_suppressor()
    
    # Step 4: Test TensorFlow
    test_success = test_tensorflow_gpu()
    
    print("\n" + "=" * 50)
    print("📋 Solusi NUMA Warning:")
    print("1. ✅ Environment variables dikonfigurasi")
    print("2. ✅ GPU information checked")
    print("3. ✅ TensorFlow config dibuat")
    print(f"4. {'✅' if test_success else '❌'} GPU test {'berhasil' if test_success else 'gagal'}")
    
    print("\n💡 Untuk menerapkan solusi:")
    print("1. Tambahkan kode TF config ke file utama")
    print("2. Set environment variables sebelum import TensorFlow")
    print("3. Restart training script")
    
    print("\n🔧 TensorFlow Configuration Code:")
    print("-" * 30)
    print(tf_config)

if __name__ == "__main__":
    main()
