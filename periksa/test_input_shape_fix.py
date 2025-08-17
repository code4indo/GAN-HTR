#!/usr/bin/env python3
"""
Test script untuk memverifikasi bahwa input_shape warnings sudah teratasi
"""

import os
import sys
import warnings
import subprocess

def test_reshape_without_input_shape():
    """
    Test layer Reshape tanpa input_shape parameter
    """
    print("🧪 Testing Reshape layer without input_shape...")
    
    try:
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Import TensorFlow/Keras
            import tensorflow as tf
            from tensorflow.keras.layers import Reshape, Input
            from tensorflow.keras.models import Model
            
            # Test Reshape without input_shape (correct way)
            input_layer = Input(shape=(128, 1024, 1))
            reshaped = Reshape((1024, 128, 1))(input_layer)
            
            print(f"✅ Reshape layer created successfully: {reshaped.shape}")
            
            # Check for input_shape warnings
            reshape_warnings = [warning for warning in w if 'input_shape' in str(warning.message).lower() or 'input_dim' in str(warning.message).lower()]
            
            if reshape_warnings:
                print("⚠️  Input_shape warnings still present:")
                for warning in reshape_warnings:
                    print(f"   - {warning.message}")
                return False
            else:
                print("✅ No input_shape warnings detected!")
                return True
                
    except Exception as e:
        print(f"❌ Error during Reshape test: {e}")
        return False

def test_sequential_model_correct_way():
    """
    Test Sequential model dengan Input layer yang benar
    """
    print("\n🧪 Testing Sequential model with correct Input layer...")
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, Dense, LeakyReLU
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Correct way: Use Input layer as first layer
            model = Sequential([
                Input(shape=(100,)),  # Use Input() layer
                Dense(64),            # No need for input_shape
                LeakyReLU(negative_slope=0.2),
                Dense(32),
                LeakyReLU(negative_slope=0.1),
                Dense(1, activation='sigmoid')
            ])
            
            print(f"✅ Sequential model created with {len(model.layers)} layers")
            
            # Check for warnings
            input_warnings = [warning for warning in w if 'input_shape' in str(warning.message).lower() or 'input_dim' in str(warning.message).lower()]
            
            if input_warnings:
                print("⚠️  Input warnings in Sequential model:")
                for warning in input_warnings:
                    print(f"   - {warning.message}")
                return False
            else:
                print("✅ Sequential model created without input warnings!")
                return True
                
    except Exception as e:
        print(f"❌ Error during Sequential model test: {e}")
        return False

def test_functional_api_model():
    """
    Test Functional API model (yang digunakan di GAN)
    """
    print("\n🧪 Testing Functional API model...")
    
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Conv2D, Reshape
        from tensorflow.keras.models import Model
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Functional API (seperti yang digunakan di GAN)
            inputs = Input(shape=(128, 1024, 1))
            
            # Some processing layers
            x = Conv2D(64, (3, 3), padding='same')(inputs)
            
            # Reshape tanpa input_shape (fixed version)
            reshaped = Reshape((1024, 128, 1))(x)
            
            # Create model
            model = Model(inputs=inputs, outputs=reshaped)
            
            print(f"✅ Functional API model created: {model.input_shape} → {model.output_shape}")
            
            # Check for warnings
            reshape_warnings = [warning for warning in w if 'input_shape' in str(warning.message).lower()]
            
            if reshape_warnings:
                print("⚠️  Reshape warnings in Functional API:")
                for warning in reshape_warnings:
                    print(f"   - {warning.message}")
                return False
            else:
                print("✅ Functional API model without Reshape warnings!")
                return True
                
    except Exception as e:
        print(f"❌ Error during Functional API test: {e}")
        return False

def verify_files_fixed():
    """
    Verifikasi bahwa file-file sudah diperbaiki
    """
    print("\n🔍 Verifying fixed files...")
    
    files_to_check = [
        "/home/lambda_one/tesis/GAN-HTR/jnm_GAN_AHTR.py",
        "/home/lambda_one/tesis/GAN-HTR/jnm_GAN_AHTR copy.py",
        "/home/lambda_one/tesis/GAN-HTR/GAN_AHTR.py",
        "/home/lambda_one/tesis/GAN-HTR/create_working_file.py",
        "/home/lambda_one/tesis/GAN-HTR/periksa/train_gan_nan.py",
        "/home/lambda_one/tesis/GAN-HTR/periksa/train_gan_optimized.py"
    ]
    
    fixed_count = 0
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if input_shape is still used in Reshape
                if 'Reshape(' in content and 'input_shape=' in content:
                    print(f"❌ {os.path.basename(file_path)}: Still has input_shape in Reshape")
                else:
                    print(f"✅ {os.path.basename(file_path)}: Fixed")
                    fixed_count += 1
                    
            except Exception as e:
                print(f"⚠️  Error checking {file_path}: {e}")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print(f"📊 Files fixed: {fixed_count}/{len(files_to_check)}")
    return fixed_count == len(files_to_check)

def main():
    """
    Main test function
    """
    print("🔍 Input Shape Warning Fix Verification")
    print("=" * 60)
    
    # Run tests
    test1 = test_reshape_without_input_shape()
    test2 = test_sequential_model_correct_way() 
    test3 = test_functional_api_model()
    test4 = verify_files_fixed()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"   ✅ Reshape test: {'PASSED' if test1 else 'FAILED'}")
    print(f"   ✅ Sequential model test: {'PASSED' if test2 else 'FAILED'}")
    print(f"   ✅ Functional API test: {'PASSED' if test3 else 'FAILED'}")
    print(f"   ✅ Files verification: {'PASSED' if test4 else 'FAILED'}")
    
    all_passed = test1 and test2 and test3 and test4
    
    if all_passed:
        print("\n🎉 All tests PASSED!")
        print("💡 Input_shape warnings successfully fixed!")
        print("\n📋 Summary of fixes:")
        print("   - Removed input_shape from Reshape layers")
        print("   - Fixed Sequential model to use Input() layer")
        print("   - Functional API models work correctly")
        print("\n🚀 Ready to run training without input_shape warnings!")
    else:
        print("\n❌ Some tests FAILED.")
        print("💡 Check the files and configuration again.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
