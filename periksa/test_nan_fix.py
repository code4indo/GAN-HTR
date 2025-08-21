#!/usr/bin/env python3
"""
Test script untuk memverifikasi fix NaN error
"""

import tensorflow as tf
import numpy as np
import sys
import os

# Add current directory to path to import fixed CTC loss
sys.path.append('/home/lambda_one/tesis/GAN-HTR')
sys.path.append('/home/lambda_one/tesis/GAN-HTR/periksa')

def test_fixed_ctc_loss():
    """Test the fixed CTC loss implementation"""
    print("🧪 Testing Fixed CTC Loss Implementation...")
    
    try:
        from periksa.fixed_ctc_loss import FixedCTCLoss
        
        # Create test data
        batch_size = 2
        max_time_steps = 100
        num_classes = 80
        max_label_length = 20
        
        # Create realistic test data
        y_pred = tf.random.normal([batch_size, max_time_steps, num_classes])
        y_pred = tf.nn.softmax(y_pred, axis=-1)  # Make it probability-like
        
        # Create labels with some padding (0s)
        y_true = tf.random.uniform([batch_size, max_label_length], 1, num_classes-1, dtype=tf.int32)
        # Add some padding
        y_true = tf.where(tf.random.uniform([batch_size, max_label_length]) > 0.7, 0, y_true)
        
        # Test fixed CTC loss
        ctc_loss = FixedCTCLoss()
        loss_value = ctc_loss.safe_ctc_loss(y_true, y_pred)
        
        # Check if loss is valid (not NaN or Inf)
        if tf.math.is_finite(loss_value):
            print(f"✅ Fixed CTC Loss computed successfully: {loss_value:.4f}")
            return True
        else:
            print(f"❌ CTC Loss still produces invalid value: {loss_value}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing CTC loss: {e}")
        return False

def test_shape_conversion():
    """Test the shape conversion for generator to CRNN"""
    print("🧪 Testing Shape Conversion...")
    
    try:
        from periksa.fixed_ctc_loss import fix_generator_to_crnn_shape
        
        # Simulate generator output
        batch_size = 2
        height = 128
        width = 1024
        channels = 1
        
        generator_out = tf.random.normal([batch_size, height, width, channels])
        crnn_input = fix_generator_to_crnn_shape(generator_out)
        
        expected_shape = [batch_size, width, height * channels]
        
        if list(crnn_input.shape) == expected_shape:
            print(f"✅ Shape conversion successful:")
            print(f"   Generator output: {generator_out.shape}")
            print(f"   CRNN input: {crnn_input.shape}")
            return True
        else:
            print(f"❌ Shape conversion failed:")
            print(f"   Expected: {expected_shape}")
            print(f"   Got: {crnn_input.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing shape conversion: {e}")
        return False

def test_basic_tensor_operations():
    """Test basic TensorFlow operations to ensure environment is OK"""
    print("🧪 Testing Basic TensorFlow Operations...")
    
    try:
        # Test basic operations
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = tf.add(a, b)
        
        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        print(f"   Available GPUs: {len(gpus)}")
        
        # Test mixed precision (should be disabled for stability)
        current_policy = tf.keras.mixed_precision.global_policy()
        print(f"   Current mixed precision policy: {current_policy.name}")
        
        if current_policy.name == 'float32':
            print("   ✅ Mixed precision properly disabled")
        else:
            print("   ⚠️ Mixed precision still enabled - may cause NaN")
        
        print("✅ Basic TensorFlow operations working")
        return True
        
    except Exception as e:
        print(f"❌ Error in basic TF operations: {e}")
        return False

def run_mini_training_simulation():
    """Run a minimal training simulation to check for NaN issues"""
    print("🧪 Running Mini Training Simulation...")
    
    try:
        # Create simple mock data
        batch_size = 2
        
        # Mock images
        mock_degraded = tf.random.normal([batch_size, 128, 1024, 1])
        mock_clean = tf.random.normal([batch_size, 128, 1024, 1])
        
        # Mock labels
        mock_labels = tf.random.uniform([batch_size, 20], 1, 79, dtype=tf.int32)
        
        # Simple generator (just pass through for testing)
        generator_out = mock_degraded + tf.random.normal([batch_size, 128, 1024, 1]) * 0.1
        
        # Test content loss
        content_loss = tf.reduce_mean(tf.square(mock_clean - generator_out))
        
        # Test shape conversion
        from periksa.fixed_ctc_loss import fix_generator_to_crnn_shape, FixedCTCLoss
        crnn_input = fix_generator_to_crnn_shape(generator_out)
        
        # Create mock CRNN output (logits)
        num_classes = 80
        crnn_logits = tf.random.normal([batch_size, crnn_input.shape[1], num_classes])
        
        # Test CTC loss
        ctc_loss = FixedCTCLoss()
        recognition_loss = ctc_loss.safe_ctc_loss(mock_labels, crnn_logits)
        
        # Combine losses
        total_loss = content_loss + recognition_loss
        
        # Check for NaN/Inf
        if tf.math.is_finite(total_loss):
            print(f"✅ Mini training simulation successful:")
            print(f"   Content loss: {content_loss:.4f}")
            print(f"   Recognition loss: {recognition_loss:.4f}")
            print(f"   Total loss: {total_loss:.4f}")
            return True
        else:
            print(f"❌ Mini training simulation failed - loss is invalid: {total_loss}")
            return False
            
    except Exception as e:
        print(f"❌ Error in mini training simulation: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running NaN Error Fix Verification Tests...\n")
    
    tests = [
        ("Basic TensorFlow Operations", test_basic_tensor_operations),
        ("Fixed CTC Loss", test_fixed_ctc_loss),
        ("Shape Conversion", test_shape_conversion),
        ("Mini Training Simulation", run_mini_training_simulation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 Running: {test_name}")
        print('='*50)
        
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary")
    print('='*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! The fix should work correctly.")
        print("🚀 Ready to apply fix with: poetry run python periksa/apply_nan_fix.py")
    else:
        print(f"\n⚠️ {len(tests) - passed} test(s) failed. Check the errors above.")
        print("🔧 You may need to adjust the fix implementation.")

if __name__ == "__main__":
    main()