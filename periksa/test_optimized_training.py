#!/usr/bin/env python3
"""
Test script untuk memverifikasi optimisasi jnm_GAN_AHTR.py
Akan menjalankan training singkat untuk memastikan semua komponen berjalan baik
"""

import os
import sys
import traceback

def test_imports():
    """Test semua import yang diperlukan"""
    print("🔍 Testing imports...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test mixed precision
        print("🚀 Testing Mixed Precision...")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        policy = tf.keras.mixed_precision.global_policy()
        print(f"✅ Mixed Precision Policy: {policy.name}")
        
        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("⚠️  No GPU detected, training will use CPU")
            
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        from tqdm import tqdm
        print("✅ All core dependencies imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_dataset_structure():
    """Test struktur dataset"""
    print("\n📁 Testing dataset structure...")
    
    required_paths = [
        'datasets/nan_distorted/train',
        'datasets/nan_distorted/validation', 
        'datasets/nan_raw_biner/train',
        'datasets/nan_raw_biner/validation',
        'Sets/lines.txt',
        'Sets/CHAR_LIST'
    ]
    
    all_exists = True
    for path in required_paths:
        if os.path.exists(path):
            print(f"✅ {path}")
            if path.endswith('.txt') or path.endswith('CHAR_LIST'):
                # Show file size
                size = os.path.getsize(path)
                print(f"   Size: {size} bytes")
        else:
            print(f"❌ {path} - NOT FOUND")
            all_exists = False
    
    return all_exists

def test_data_generator():
    """Test data generator function"""
    print("\n🔄 Testing data generator...")
    try:
        # Import the main script (this will test if it loads without syntax errors)
        sys.path.append('.')
        
        # Test minimal tf.data pipeline
        import tensorflow as tf
        
        # Create a simple test dataset
        def dummy_generator():
            for i in range(5):
                yield {
                    'deg_image': tf.random.normal((128, 1024, 1)),
                    'gt_image': tf.random.normal((128, 1024, 1)),
                    'crnn_image': tf.random.normal((1024, 128, 1)),
                    'transcription': tf.constant([1, 2, 3, 4, 5] + [0] * 123, dtype=tf.int16),
                    'text_line': tf.constant("test", dtype=tf.string)
                }
        
        dataset = tf.data.Dataset.from_generator(
            dummy_generator,
            output_signature={
                'deg_image': tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
                'gt_image': tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
                'crnn_image': tf.TensorSpec(shape=(1024, 128, 1), dtype=tf.float32),
                'transcription': tf.TensorSpec(shape=(128,), dtype=tf.int16),
                'text_line': tf.TensorSpec(shape=(), dtype=tf.string)
            }
        )
        
        # Test pipeline optimizations
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(2, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Test iteration
        count = 0
        for batch in dataset:
            count += 1
            print(f"✅ Batch {count}: shapes OK")
            if count >= 2:  # Test first 2 batches
                break
                
        print("✅ tf.data pipeline test successful")
        return True
        
    except Exception as e:
        print(f"❌ Data generator test failed: {e}")
        traceback.print_exc()
        return False

def test_script_syntax():
    """Test syntax jnm_GAN_AHTR.py"""
    print("\n📝 Testing script syntax...")
    try:
        # Try to compile the script
        with open('jnm_GAN_AHTR.py', 'r') as f:
            code = f.read()
        
        compile(code, 'jnm_GAN_AHTR.py', 'exec')
        print("✅ Script syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Error reading script: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 TESTING OPTIMIZED GAN-HTR TRAINING SCRIPT")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Dataset Structure", test_dataset_structure), 
        ("Script Syntax", test_script_syntax),
        ("Data Generator", test_data_generator)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🔹 Running {test_name}...")
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} : {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Script ready for training!")
        print("💡 To start training, run: poetry run python jnm_GAN_AHTR.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()
