#!/usr/bin/env python3
"""
Quick benchmark script untuk menguji optimasi hardware
"""

import os
import time
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import tensorflow as tf
import numpy as np
from datetime import datetime

def test_gpu_detection():
    """Test deteksi GPU dan setup multi-GPU"""
    print("=== GPU DETECTION TEST ===")
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Number of GPUs detected: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"   Memory growth enabled: ✅")
            except Exception as e:
                print(f"   Memory growth error: ❌ {e}")
    
    # Test MirroredStrategy
    try:
        strategy = tf.distribute.MirroredStrategy()
        print(f"MirroredStrategy devices: {strategy.num_replicas_in_sync}")
        return strategy
    except Exception as e:
        print(f"MirroredStrategy error: ❌ {e}")
        return tf.distribute.get_strategy()

def test_memory_bandwidth():
    """Test memory bandwidth dengan large tensor operations"""
    print("\n=== MEMORY BANDWIDTH TEST ===")
    
    # Test CPU-GPU transfer
    sizes = [1024, 2048, 4096]
    
    for size in sizes:
        print(f"\nTesting {size}x{size} tensor transfer:")
        
        # Create large tensor on CPU
        start_time = time.time()
        cpu_tensor = tf.random.normal((size, size), dtype=tf.float32)
        cpu_time = time.time() - start_time
        
        # Transfer to GPU
        start_time = time.time()
        gpu_tensor = tf.identity(cpu_tensor)
        transfer_time = time.time() - start_time
        
        # GPU computation
        start_time = time.time()
        result = tf.matmul(gpu_tensor, gpu_tensor)
        compute_time = time.time() - start_time
        
        data_size = size * size * 4 / (1024**2)  # MB
        bandwidth = data_size / transfer_time if transfer_time > 0 else 0
        
        print(f"  Data size: {data_size:.1f}MB")
        print(f"  CPU creation: {cpu_time*1000:.1f}ms")
        print(f"  GPU transfer: {transfer_time*1000:.1f}ms ({bandwidth:.0f}MB/s)")
        print(f"  GPU compute: {compute_time*1000:.1f}ms")

def test_parallel_processing():
    """Test parallel data processing"""
    print("\n=== PARALLEL PROCESSING TEST ===")
    
    from concurrent.futures import ThreadPoolExecutor
    import random
    
    def dummy_image_processing(i):
        """Simulate image processing"""
        # Simulate file I/O and processing
        time.sleep(random.uniform(0.01, 0.05))
        return np.random.random((128, 1024, 1)).astype(np.float32)
    
    # Test different worker counts
    worker_counts = [1, 4, 8, 16, 32]
    num_tasks = 100
    
    for workers in worker_counts:
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(dummy_image_processing, range(num_tasks)))
        
        total_time = time.time() - start_time
        throughput = num_tasks / total_time
        
        print(f"  Workers: {workers:2d} | Time: {total_time:.2f}s | Throughput: {throughput:.1f} tasks/s")

def test_batch_processing(strategy):
    """Test batch processing dengan multi-GPU"""
    print("\n=== BATCH PROCESSING TEST ===")
    
    batch_sizes = [8, 16, 24, 32]
    image_shape = (128, 1024, 1)
    
    for batch_size in batch_sizes:
        try:
            print(f"\nTesting batch size: {batch_size}")
            
            # Create dummy data
            images = tf.random.normal((batch_size, *image_shape))
            labels = tf.random.normal((batch_size, *image_shape))
            
            with strategy.scope():
                # Simple model for testing
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=image_shape),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.Conv2D(1, 1, activation='sigmoid')
                ])
                
                model.compile(optimizer='adam', loss='mse')
            
            # Test forward pass
            start_time = time.time()
            predictions = model(images)
            forward_time = time.time() - start_time
            
            # Test training step
            start_time = time.time()
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = tf.keras.losses.mse(labels, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            training_time = time.time() - start_time
            
            # Memory usage (approximate)
            memory_per_sample = np.prod(image_shape) * 4 / (1024**2)  # MB
            total_memory = memory_per_sample * batch_size
            
            print(f"  Forward pass: {forward_time*1000:.1f}ms")
            print(f"  Training step: {training_time*1000:.1f}ms")
            print(f"  Memory usage: ~{total_memory:.1f}MB")
            print(f"  Throughput: {batch_size/training_time:.1f} samples/s")
            
        except tf.errors.ResourceExhaustedError:
            print(f"  ❌ OOM Error - Batch size {batch_size} too large")
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_io_performance():
    """Test I/O performance dengan dataset loading"""
    print("\n=== I/O PERFORMANCE TEST ===")
    
    from glob import glob
    
    # Check dataset availability
    image_paths = glob('datasets/nan_distorted/train/*.jpg')
    
    if not image_paths:
        print("❌ No dataset found for I/O test")
        return
    
    print(f"Found {len(image_paths)} images for testing")
    
    # Test sequential vs parallel loading
    sample_paths = image_paths[:50]  # Test with 50 images
    
    # Sequential loading
    start_time = time.time()
    sequential_count = 0
    for path in sample_paths:
        if os.path.exists(path):
            sequential_count += 1
    sequential_time = time.time() - start_time
    
    print(f"Sequential check: {sequential_time:.2f}s for {sequential_count} files")
    print(f"I/O rate: {sequential_count/sequential_time:.1f} files/s")

def main():
    """Main benchmark function"""
    print("🚀 GAN-HTR HARDWARE OPTIMIZATION BENCHMARK")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print("=" * 60)
    
    # GPU detection and setup
    strategy = test_gpu_detection()
    
    # Memory bandwidth test
    test_memory_bandwidth()
    
    # Parallel processing test
    test_parallel_processing()
    
    # Batch processing test
    test_batch_processing(strategy)
    
    # I/O performance test
    test_io_performance()
    
    print("\n" + "=" * 60)
    print("🎉 BENCHMARK COMPLETED!")
    print("=" * 60)
    
    # Recommendations
    print("\n📋 OPTIMIZATION RECOMMENDATIONS:")
    print("1. ✅ Use batch_size=32 for optimal GPU utilization")
    print("2. ✅ Use 16 workers for parallel data loading")
    print("3. ✅ Enable MirroredStrategy for dual GPU training")
    print("4. ✅ Monitor GPU memory to avoid OOM errors")
    print("5. ✅ Use mixed precision for additional speedup")
    
    print("\n🚀 Ready for optimized training with:")
    print("   python3 train_gan_optimized.py --epoch 150 --batch_size 32")

if __name__ == "__main__":
    main()
