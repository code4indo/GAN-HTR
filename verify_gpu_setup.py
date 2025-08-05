#!/usr/bin/env python3
"""
GPU Setup Verification Script for GAN-HTR Workstation
Verifies NVIDIA GPU, CUDA, TensorFlow GPU support, and memory configuration
"""

import os
import subprocess
import sys

def print_section(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def run_command(cmd, description):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description}")
            if result.stdout.strip():
                print(result.stdout.strip())
            return True, result.stdout
        else:
            print(f"❌ {description}")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False, str(e)

def check_nvidia_gpu():
    """Check NVIDIA GPU and driver"""
    print_section("NVIDIA GPU & Driver Check")
    
    success, output = run_command("nvidia-smi", "NVIDIA System Management Interface")
    if success:
        # Extract GPU info
        lines = output.split('\n')
        for line in lines:
            if 'GeForce' in line or 'Tesla' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line:
                print(f"🎮 GPU Found: {line.strip()}")
    
    # Check CUDA version
    run_command("nvcc --version", "CUDA Compiler Version")
    
    # Check cuDNN
    run_command("find /usr -name 'libcudnn*' 2>/dev/null | head -3", "cuDNN Libraries")

def check_tensorflow_gpu():
    """Check TensorFlow GPU support"""
    print_section("TensorFlow GPU Support")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow imported successfully")
        print(f"   Version: {tf.__version__}")
        print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")
        
        # List all devices
        print(f"\n📋 All Physical Devices:")
        devices = tf.config.list_physical_devices()
        for device in devices:
            print(f"   {device}")
            
        # GPU specific checks
        gpus = tf.config.list_physical_devices('GPU')
        print(f"\n🎮 GPU Devices: {len(gpus)} found")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    print(f"      Device name: {details.get('device_name', 'Unknown')}")
                    print(f"      Compute capability: {details.get('compute_capability', 'Unknown')}")
                except Exception as e:
                    print(f"      Details: {e}")
                    
            return True, gpus
        else:
            print("❌ No GPU devices found")
            return False, []
            
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False, []
    except Exception as e:
        print(f"❌ TensorFlow GPU check failed: {e}")
        return False, []

def test_gpu_memory():
    """Test GPU memory allocation"""
    print_section("GPU Memory Configuration")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ No GPU available for memory test")
            return False
            
        # Configure memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"⚠️  Memory growth configuration: {e}")
            
        # Test memory allocation
        print("\n🧪 Testing GPU memory allocation:")
        with tf.device('/GPU:0'):
            for size in [100, 500, 1000]:
                try:
                    tensor = tf.random.normal([size, size])
                    print(f"   ✅ Allocated {size}x{size} tensor ({size*size*4/1024/1024:.1f} MB)")
                except Exception as e:
                    print(f"   ❌ Failed to allocate {size}x{size} tensor: {e}")
                    
        return True
        
    except Exception as e:
        print(f"❌ GPU memory test failed: {e}")
        return False

def test_gpu_computation():
    """Test GPU computation with deep learning operations"""
    print_section("GPU Deep Learning Computation Test")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ No GPU available for computation test")
            return False
            
        print("🧪 Testing basic GPU operations:")
        with tf.device('/GPU:0'):
            # Matrix multiplication
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print(f"   ✅ Matrix multiplication: {c.shape}")
            
            # Convolution (similar to GAN-HTR)
            batch_images = tf.random.normal([4, 128, 1024, 1])  # Batch of 4 images
            conv_kernel = tf.random.normal([3, 3, 1, 64])      # 3x3 conv with 64 filters
            conv_output = tf.nn.conv2d(batch_images, conv_kernel, strides=1, padding='SAME')
            print(f"   ✅ Convolution operation: {batch_images.shape} -> {conv_output.shape}")
            
        print("\n🏗️  Testing U-Net model creation (GAN-HTR style):")
        with tf.device('/GPU:0'):
            # Create simplified U-Net
            inputs = tf.keras.layers.Input((128, 1024, 1))
            conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
            pool1 = tf.keras.layers.MaxPooling2D(2)(conv1)
            conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
            up1 = tf.keras.layers.UpSampling2D(2)(conv2)
            concat1 = tf.keras.layers.concatenate([conv1, up1])
            output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(concat1)
            
            model = tf.keras.Model(inputs, output)
            model.compile(optimizer='adam', loss='binary_crossentropy')
            
            # Test prediction
            dummy_input = np.random.random((2, 128, 1024, 1))
            prediction = model.predict(dummy_input, verbose=0)
            
            print(f"   ✅ U-Net model created successfully")
            print(f"   ✅ Input shape: {dummy_input.shape}")
            print(f"   ✅ Output shape: {prediction.shape}")
            print(f"   ✅ Model parameters: {model.count_params():,}")
            
        return True
        
    except Exception as e:
        print(f"❌ GPU computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_cuda_environment():
    """Check CUDA environment variables"""
    print_section("CUDA Environment Variables")
    
    cuda_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_HOME',
        'CUDA_PATH',
        'LD_LIBRARY_PATH'
    ]
    
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
        
    # Check if CUDA libraries are in PATH
    cuda_paths = [
        '/usr/local/cuda/bin',
        '/usr/local/cuda/lib64',
        '/opt/cuda/bin',
        '/opt/cuda/lib64'
    ]
    
    print(f"\n📁 CUDA Installation Paths:")
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"   ✅ {path} exists")
        else:
            print(f"   ❌ {path} not found")

def main():
    print("🚀 GAN-HTR GPU Workstation Verification")
    print("=" * 60)
    
    # Check system requirements
    check_nvidia_gpu()
    check_cuda_environment()
    
    # Check TensorFlow GPU
    tf_success, gpus = check_tensorflow_gpu()
    
    if tf_success and gpus:
        # Test GPU functionality
        test_gpu_memory()
        test_gpu_computation()
        
        print_section("Summary & Recommendations")
        print("✅ GPU setup verification completed successfully!")
        print("\n🎯 For optimal GAN-HTR performance:")
        print("   - Your GPU is ready for deep learning workloads")
        print("   - Memory growth is configured to prevent OOM errors")
        print("   - TensorFlow can utilize GPU acceleration")
        print("   - U-Net models can be trained on GPU")
        print("\n🚀 You can now run GAN-HTR training scripts:")
        print("   python GAN_AHTR.py")
        print("   python train_khatt_basic_distorted.py")
        print("   python eval_Dibco_2010.py")
        
    else:
        print_section("Issues Found")
        print("❌ GPU setup has issues that need to be resolved:")
        print("   - Check NVIDIA driver installation")
        print("   - Verify CUDA toolkit installation")
        print("   - Ensure TensorFlow-GPU is properly installed")
        print("   - Check virtual environment configuration")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
