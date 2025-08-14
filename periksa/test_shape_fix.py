#!/usr/bin/env python3
"""
Simple test untuk shape mismatch error
"""

import os
import sys
import tensorflow as tf

# Set environment
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable Mixed Precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def test_shape_fix():
    """Test fix untuk shape mismatch"""
    print("🧪 Testing Shape Fix")
    print("=" * 30)
    
    try:
        # Import module
        import jnm_GAN_AHTR as gan_script
        
        print("✅ Script imported successfully")
        
        # Test the data generator directly
        list_image_train = gan_script.read_file_shuffle(gan_script.rootPath + 'Sets/list_train_nan.txt')
        list_lines = gan_script.read_file(gan_script.rootPath + 'Sets/lines.txt')
        
        print(f"✅ Found {len(list_image_train)} training images")
        print(f"✅ Found {len(list_lines)} transcription lines")
        
        # Test generator with small batch
        print("🔍 Testing data generator...")
        
        generator_func = lambda: gan_script.data_generator(list_image_train[:5], list_lines, 'train')
        
        dataset = tf.data.Dataset.from_generator(
            generator_func,
            output_signature={
                'deg_image': tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
                'gt_image': tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
                'crnn_image': tf.TensorSpec(shape=(1024, 128, 1), dtype=tf.float32),
                'transcription': tf.TensorSpec(shape=(gan_script.max_text_length,), dtype=tf.int16),
                'text_line': tf.TensorSpec(shape=(), dtype=tf.string)
            }
        )
        
        dataset = dataset.batch(2, drop_remainder=True)
        
        # Test iteration
        for i, batch in enumerate(dataset.take(1)):
            print(f"✅ Batch {i+1} shapes:")
            print(f"   deg_image: {batch['deg_image'].shape}")
            print(f"   gt_image: {batch['gt_image'].shape}")
            print(f"   crnn_image: {batch['crnn_image'].shape}")
            print(f"   transcription: {batch['transcription'].shape}")
            
        print("🎉 Shape fix successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_shape_fix()
