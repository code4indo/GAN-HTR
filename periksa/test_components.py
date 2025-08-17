#!/usr/bin/env python3
"""
COMPONENT TESTING SCRIPT
Test individual components untuk isolate masalah training

Usage: poetry run python periksa/test_components.py
"""

import os
import sys
sys.path.append('/home/lambda_one/tesis/GAN-HTR')

# Set environment
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from glob import glob

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU configured: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"⚠️ GPU setup warning: {e}")

def test_imports():
    """Test importing semua modules yang diperlukan"""
    print("🧪 Testing imports...")
    
    try:
        from jnm_GAN_AHTR import (
            unet, build_discriminator_1, build_discriminator_2,
            read_file, read_file_shuffle, normalizeTranscription,
            encode_txt, charset_base, max_text_length,
            rootPath, DatabasePath, readGrayPair
        )
        from data import preproc as pp
        print("✅ All imports successful")
        return True, locals()
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False, {}

def test_dataset_paths():
    """Test keberadaan dataset dan file paths"""
    print("\n🧪 Testing dataset paths...")
    
    from jnm_GAN_AHTR import rootPath, DatabasePath
    
    paths_to_check = [
        (rootPath + 'Sets/list_train_nan.txt', 'Training list'),
        (rootPath + 'Sets/list_valid_nan.txt', 'Validation list'),
        (rootPath + 'Sets/lines.txt', 'Lines transcription'),
        (rootPath + 'charlist.txt', 'Character list'),
        (DatabasePath, 'Database directory')
    ]
    
    all_good = True
    for path, desc in paths_to_check:
        if os.path.exists(path):
            print(f"✅ {desc}: {path}")
        else:
            print(f"❌ {desc} NOT FOUND: {path}")
            all_good = False
    
    return all_good

def test_character_encoding():
    """Test character encoding dan charset"""
    print("\n🧪 Testing character encoding...")
    
    try:
        from jnm_GAN_AHTR import charset_base, encode_txt, normalizeTranscription
        
        print(f"✅ Charset loaded: {len(charset_base)} characters")
        print(f"   Sample chars: {charset_base[:10]}")
        
        # Test encoding
        test_text = "hello world"
        normalized = normalizeTranscription(test_text)
        encoded = encode_txt(normalized)
        
        print(f"✅ Text encoding test:")
        print(f"   Original: '{test_text}'")
        print(f"   Normalized: '{normalized}'")
        print(f"   Encoded: {encoded}")
        
        if len(encoded) > 0:
            print("✅ Character encoding working")
            return True
        else:
            print("❌ Character encoding failed")
            return False
            
    except Exception as e:
        print(f"❌ Character encoding test failed: {e}")
        return False

def test_data_loading():
    """Test loading individual images dan transcriptions"""
    print("\n🧪 Testing data loading...")
    
    try:
        from jnm_GAN_AHTR import (
            read_file_shuffle, read_file, readGrayPair,
            rootPath, DatabasePath
        )
        
        # Load file lists
        list_image_train = read_file_shuffle(rootPath + 'Sets/list_train_nan.txt')
        list_lines = read_file(rootPath + 'Sets/lines.txt')
        
        print(f"✅ Loaded {len(list_image_train)} training images")
        print(f"✅ Loaded {len(list_lines)} transcription lines")
        
        # Test loading first few images
        success_count = 0
        for i, im_base in enumerate(list_image_train[:5]):
            try:
                # Find actual file
                search_pattern = os.path.join('datasets/nan_distorted', 'train', im_base + '.*')
                found_files = glob(search_pattern)
                
                if not found_files:
                    print(f"⚠️ Image not found: {im_base}")
                    continue
                
                im_full_name = os.path.basename(found_files[0])
                
                # Load images
                deg_image, gt_image = readGrayPair(im_full_name, split='train')
                
                print(f"✅ Image {i+1}: {im_base}")
                print(f"   Degraded: {deg_image.shape}, range: [{deg_image.min():.3f}, {deg_image.max():.3f}]")
                print(f"   GT: {gt_image.shape}, range: [{gt_image.min():.3f}, {gt_image.max():.3f}]")
                
                # Check for NaN
                if np.any(np.isnan(deg_image)) or np.any(np.isnan(gt_image)):
                    print(f"❌ NaN detected in {im_base}")
                else:
                    success_count += 1
                    
            except Exception as e:
                print(f"❌ Error loading {im_base}: {e}")
        
        if success_count > 0:
            print(f"✅ Data loading test passed: {success_count}/5 successful")
            return True
        else:
            print("❌ Data loading test failed: no successful loads")
            return False
            
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_model_creation():
    """Test creating all models"""
    print("\n🧪 Testing model creation...")
    
    try:
        from jnm_GAN_AHTR import unet, build_discriminator_1, build_discriminator_2
        
        # Test generator
        print("   Creating generator...")
        generator = unet()
        print(f"✅ Generator created: input {generator.input_shape}, output {generator.output_shape}")
        
        # Test discriminator 1
        print("   Creating discriminator 1...")
        discriminator_1 = build_discriminator_1()
        print(f"✅ Discriminator 1 created: inputs {[inp.shape for inp in discriminator_1.inputs]}")
        
        # Test discriminator 2 (CRNN)
        print("   Creating discriminator 2 (CRNN)...")
        discriminator_2 = build_discriminator_2()
        print(f"✅ Discriminator 2 created: input {discriminator_2.input_shape}, output {discriminator_2.output_shape}")
        
        print("✅ All models created successfully")
        return True, (generator, discriminator_1, discriminator_2)
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_forward_pass():
    """Test forward pass melalui semua models"""
    print("\n🧪 Testing forward pass...")
    
    try:
        # Create models
        success, models = test_model_creation()
        if not success:
            return False
        
        generator, discriminator_1, discriminator_2 = models
        
        # Create dummy data
        batch_size = 1
        dummy_deg = np.random.random((batch_size, 128, 1024, 1)).astype(np.float32)
        dummy_gt = np.random.random((batch_size, 128, 1024, 1)).astype(np.float32)
        dummy_crnn = np.random.random((batch_size, 1024, 128, 1)).astype(np.float32)
        
        print("   Testing generator forward pass...")
        gen_output = generator(dummy_deg, training=False)
        print(f"✅ Generator output: {gen_output.shape}")
        
        if np.any(np.isnan(gen_output)):
            print("❌ Generator output contains NaN")
            return False
        
        print("   Testing discriminator 1 forward pass...")
        disc1_output = discriminator_1([dummy_gt, dummy_deg], training=False)
        print(f"✅ Discriminator 1 output: {disc1_output.shape}")
        
        if np.any(np.isnan(disc1_output)):
            print("❌ Discriminator 1 output contains NaN")
            return False
        
        print("   Testing discriminator 2 (CRNN) forward pass...")
        disc2_output = discriminator_2(dummy_crnn, training=False)
        print(f"✅ Discriminator 2 output: {disc2_output.shape}")
        
        if np.any(np.isnan(disc2_output)):
            print("❌ Discriminator 2 output contains NaN")
            return False
        
        print("✅ All forward passes successful")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ctc_loss():
    """Test CTC loss function"""
    print("\n🧪 Testing CTC loss...")
    
    try:
        from jnm_GAN_AHTR import charset_base
        
        # Create sample data
        batch_size = 2
        seq_length = 256
        vocab_size = len(charset_base) + 1
        max_label_length = 20
        
        print(f"   Vocab size: {vocab_size}")
        print(f"   Sequence length: {seq_length}")
        print(f"   Max label length: {max_label_length}")
        
        # Sample predictions (logits)
        y_pred = tf.random.normal((batch_size, seq_length, vocab_size))
        print(f"✅ Created predictions: {y_pred.shape}")
        
        # Sample labels
        y_true = tf.random.uniform((batch_size, max_label_length), 
                                 maxval=vocab_size-1, dtype=tf.int32)
        print(f"✅ Created labels: {y_true.shape}")
        
        # Test original CTC loss
        print("   Testing original CTC loss...")
        try:
            from jnm_GAN_AHTR import ctc_loss_lambda_func
            loss_original = ctc_loss_lambda_func(y_true, y_pred)
            print(f"✅ Original CTC loss: {float(loss_original):.4f}")
            
            if tf.math.is_finite(loss_original):
                print("✅ Original CTC loss is finite")
            else:
                print("❌ Original CTC loss is not finite")
                
        except Exception as e:
            print(f"❌ Original CTC loss failed: {e}")
        
        # Test improved CTC loss
        print("   Testing improved CTC loss...")
        try:
            from periksa.emergency_training import UltraSafeCTCLoss
            safe_ctc = UltraSafeCTCLoss()
            loss_safe = safe_ctc.safe_ctc_loss(y_true, y_pred)
            print(f"✅ Safe CTC loss: {float(loss_safe):.4f}")
            
            if tf.math.is_finite(loss_safe):
                print("✅ Safe CTC loss is finite")
                return True
            else:
                print("❌ Safe CTC loss is not finite")
                return False
                
        except Exception as e:
            print(f"❌ Safe CTC loss failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ CTC loss test failed: {e}")
        return False

def test_memory_usage():
    """Test GPU memory usage"""
    print("\n🧪 Testing GPU memory usage...")
    
    try:
        # Get GPU info
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            print("⚠️ No GPU detected")
            return True
        
        for i, gpu in enumerate(gpus):
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"✅ GPU {i}: {gpu_details.get('device_name', 'Unknown')}")
        
        # Test memory allocation
        print("   Testing memory allocation...")
        
        # Create some tensors
        test_tensor = tf.random.normal((100, 100, 100))
        result = tf.reduce_sum(test_tensor)
        
        print(f"✅ Memory test passed: {float(result):.2f}")
        
        # Clean up
        del test_tensor, result
        tf.keras.backend.clear_session()
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_complete_pipeline():
    """Test complete pipeline dengan real data"""
    print("\n🧪 Testing complete pipeline...")
    
    try:
        from jnm_GAN_AHTR import (
            read_file_shuffle, read_file, readGrayPair,
            normalizeTranscription, encode_txt,
            rootPath, DatabasePath
        )
        from data import preproc as pp
        
        # Load one real sample
        list_image_train = read_file_shuffle(rootPath + 'Sets/list_train_nan.txt')
        list_lines = read_file(rootPath + 'Sets/lines.txt')
        
        # Find a valid sample
        for im_base in list_image_train[:10]:
            try:
                # Find image file
                search_pattern = os.path.join('datasets/nan_distorted', 'train', im_base + '.*')
                found_files = glob(search_pattern)
                
                if not found_files:
                    continue
                
                im_full_name = os.path.basename(found_files[0])
                
                # Load images
                deg_image, gt_image = readGrayPair(im_full_name, split='train')
                
                # Find transcription
                line_text = None
                for line in list_lines:
                    if line.startswith(im_full_name):
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            line_text = parts[1]
                            break
                
                if not line_text:
                    continue
                
                # Process transcription
                normalized = normalizeTranscription(line_text)
                encoded = encode_txt(normalized)
                
                if not encoded:
                    continue
                
                # Prepare CRNN data
                gt_path = os.path.join(DatabasePath, 'train', 'images', im_full_name)
                crnn_img = pp.preprocess(gt_path, (1024, 128, 1))
                
                if len(crnn_img.shape) == 2:
                    crnn_img = crnn_img.T
                    crnn_img = crnn_img[..., np.newaxis]
                elif len(crnn_img.shape) == 3 and crnn_img.shape == (128, 1024, 1):
                    crnn_img = np.transpose(crnn_img, (1, 0, 2))
                
                print(f"✅ Pipeline test with {im_base}:")
                print(f"   Degraded image: {deg_image.shape}")
                print(f"   GT image: {gt_image.shape}")
                print(f"   CRNN image: {crnn_img.shape}")
                print(f"   Text: '{line_text[:50]}...'")
                print(f"   Normalized: '{normalized[:50]}...'")
                print(f"   Encoded length: {len(encoded)}")
                
                # Test with models
                success, models = test_model_creation()
                if success:
                    generator, discriminator_1, discriminator_2 = models
                    
                    # Test generator
                    gen_out = generator(deg_image.reshape(1, 128, 1024, 1), training=False)
                    print(f"   Generator output: {gen_out.shape}")
                    
                    # Test discriminator 1
                    disc1_out = discriminator_1([gt_image.reshape(1, 128, 1024, 1), 
                                               deg_image.reshape(1, 128, 1024, 1)], training=False)
                    print(f"   Discriminator 1 output: {disc1_out.shape}")
                    
                    # Test discriminator 2
                    disc2_out = discriminator_2(crnn_img.reshape(1, 1024, 128, 1), training=False)
                    print(f"   Discriminator 2 output: {disc2_out.shape}")
                
                print("✅ Complete pipeline test successful")
                return True
                
            except Exception as e:
                print(f"⚠️ Error with {im_base}: {e}")
                continue
        
        print("❌ No valid samples found for pipeline test")
        return False
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run semua tests dan generate report"""
    
    print("🧪 COMPREHENSIVE COMPONENT TESTING")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Dataset Paths", test_dataset_paths),
        ("Character Encoding", test_character_encoding),
        ("Data Loading", test_data_loading),
        ("Model Creation", lambda: test_model_creation()[0]),
        ("Forward Pass", test_forward_pass),
        ("CTC Loss", test_ctc_loss),
        ("Memory Usage", test_memory_usage),
        ("Complete Pipeline", test_complete_pipeline)
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name}: EXCEPTION - {e}")
    
    # Summary report
    print(f"\n🏁 TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(tests) - passed}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    
    print(f"\n📊 DETAILED RESULTS:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20}: {status}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if results.get("Data Loading", False) and results.get("Model Creation", False):
        print("   ✅ Basic components working - try emergency training")
    else:
        print("   🔧 Fix failing components before attempting training")
    
    if not results.get("CTC Loss", False):
        print("   🚨 CTC Loss issues detected - use improved CTC loss")
    
    if not results.get("Memory Usage", False):
        print("   💾 Memory issues detected - reduce batch size")
    
    # Save results (only serializable data)
    import json
    results_path = "periksa/component_test_results.json"
    
    # Convert results to serializable format (only boolean values)
    serializable_results = {}
    for test_name, result in results.items():
        serializable_results[test_name] = bool(result)
    
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(tests),
            'passed': passed,
            'failed': len(tests) - passed,
            'success_rate': passed/len(tests)*100,
            'detailed_results': serializable_results
        }, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_path}")
    
    return results

def main():
    """Main function"""
    try:
        results = run_all_tests()
        
        if results.get("Complete Pipeline", False):
            print("\n🎉 All tests passed! Ready for training.")
            print("\nNext steps:")
            print("1. Run: poetry run python periksa/emergency_training.py")
            print("2. Monitor for NaN values during training")
            print("3. Gradually increase batch size if successful")
        else:
            print("\n⚠️ Some tests failed. Address issues before training.")
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
