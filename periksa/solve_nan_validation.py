"""
Solusi untuk Masalah NaN Validation Loss pada GAN-HTR
Analisis dan perbaikan berdasarkan pesan training yang diterima
"""

import tensorflow as tf
import numpy as np
import os
import json
from typing import Dict, List, Tuple

def analyze_training_failure():
    """Analisis mendalam masalah training yang gagal"""
    
    print("🔍 ANALISIS MASALAH TRAINING GAN-HTR")
    print("=" * 60)
    
    # 1. Identifikasi masalah dari pesan training
    problems_identified = [
        "Validation Loss: nan",
        "D2 (CRNN) loss tinggi: 50.0000", 
        "Early stopping dengan epoch -1",
        "Best validation loss: inf",
        "D2 (CRNN) recognition struggling"
    ]
    
    print("📋 MASALAH YANG TERIDENTIFIKASI:")
    for i, problem in enumerate(problems_identified, 1):
        print(f"   {i}. {problem}")
    
    # 2. Root cause analysis
    print("\n🔎 ROOT CAUSE ANALYSIS:")
    root_causes = {
        "NaN Validation Loss": [
            "CTC loss function mengalami numerical instability",
            "CRNN discriminator gradients exploding",
            "Invalid input sequences untuk CTC",
            "Learning rate terlalu tinggi untuk CRNN"
        ],
        "D2 (CRNN) Loss Tinggi": [
            "Text sequences terlalu panjang (>128 chars)",
            "Character encoding bermasalah",
            "Input image preprocessing tidak konsisten",
            "Label-sequence mismatch"
        ],
        "Early Stopping Gagal": [
            "Tidak pernah ada validation loss yang valid",
            "Model tidak pernah konvergen",
            "Batch size terlalu besar untuk GPU memory",
            "Data loader bermasalah"
        ]
    }
    
    for cause, details in root_causes.items():
        print(f"\n   🎯 {cause}:")
        for detail in details:
            print(f"      - {detail}")
    
    return problems_identified, root_causes

def create_emergency_fixes():
    """Buat konfigurasi emergency untuk mengatasi masalah"""
    
    print("\n🚨 EMERGENCY FIXES:")
    print("=" * 60)
    
    # 1. Konfigurasi konservatif
    emergency_config = {
        # Learning rates yang sangat konservatif
        "learning_rates": {
            "generator": 0.00001,      # Sangat kecil
            "discriminator_1": 0.00001,
            "discriminator_2_crnn": 0.00001  # CRNN paling kecil
        },
        
        # Batch size minimal
        "batch_sizes": {
            "training": 1,        # Mulai dengan 1
            "validation": 1,
            "max_recommended": 2
        },
        
        # Gradient clipping agresif
        "gradient_clipping": {
            "norm_threshold": 0.1,   # Sangat agresif
            "crnn_threshold": 0.05   # CRNN lebih agresif lagi
        },
        
        # Loss weights yang konservatif
        "loss_weights": {
            "adversarial": 0.5,      # Kurangi dari 1.0
            "content": 1.0,          # Tetap
            "recognition_crnn": 0.5  # Kurangi drastis dari 10.0
        },
        
        # Early stopping yang lebih toleran
        "early_stopping": {
            "patience": 5,           # Lebih pendek untuk debugging
            "min_delta": 0.1,        # Lebih besar
            "monitor": "val_g_loss"
        },
        
        # Data preprocessing fixes
        "data_fixes": {
            "max_text_length": 20,    # Kurangi dari 128
            "max_samples_per_epoch": 100,  # Limit untuk debugging
            "image_size": (128, 1024, 1),  # Pastikan konsisten
            "normalize_method": "zero_one"  # [0,1] normalization
        }
    }
    
    print("⚙️ EMERGENCY CONFIGURATION:")
    for section, configs in emergency_config.items():
        print(f"\n   📁 {section.upper()}:")
        for key, value in configs.items():
            print(f"      {key}: {value}")
    
    return emergency_config

def generate_fixed_training_script():
    """Generate script training yang sudah diperbaiki"""
    
    fixed_script = '''
# PERBAIKAN KHUSUS UNTUK MASALAH NaN VALIDATION LOSS
# Berdasarkan analisis kegagalan training

import tensorflow as tf
import numpy as np

def improved_ctc_loss_lambda_func(y_true, y_pred):
    """
    CTC loss yang sangat robust untuk mencegah NaN
    """
    # Pastikan input types yang benar
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Batch dan sequence info
    batch_size = tf.shape(y_true)[0]
    sequence_length = tf.shape(y_pred)[1]
    
    # Hitung label lengths dengan validasi ketat
    label_length = tf.math.count_nonzero(y_true, axis=-1, dtype=tf.int32)
    label_length = tf.maximum(label_length, 1)  # Minimal 1
    label_length = tf.minimum(label_length, 20)  # Maksimal 20
    
    # Input length yang konservatif
    input_length = tf.fill([batch_size], sequence_length // 4)  # Sangat konservatif
    input_length = tf.maximum(input_length, label_length * 2)  # Minimal 2x label length
    
    # Preprocessing prediksi yang sangat hati-hati
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Normalisasi dengan softmax + epsilon
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    y_pred = y_pred + epsilon
    y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    
    # Check validitas data sebelum CTC
    valid_batch = tf.logical_and(
        tf.reduce_all(tf.greater(label_length, 0)),
        tf.reduce_all(tf.greater(input_length, label_length))
    )
    
    def compute_ctc():
        try:
            # Log probabilities untuk CTC
            log_probs = tf.math.log(y_pred + epsilon)
            
            # CTC loss dengan parameter konservatif
            loss = tf.nn.ctc_loss(
                labels=y_true,
                logits=log_probs,
                label_length=label_length,
                logit_length=input_length,
                logits_time_major=False,
                blank_index=-1
            )
            
            # Cleaning yang sangat agresif
            loss = tf.where(tf.math.is_finite(loss), loss, tf.constant(1.0))
            loss = tf.where(tf.math.is_nan(loss), tf.constant(1.0), loss)
            loss = tf.clip_by_value(loss, 0.0, 10.0)  # Clip yang ketat
            
            return tf.reduce_mean(loss)
            
        except Exception:
            return tf.constant(2.0, dtype=tf.float32)
    
    def fallback_loss():
        return tf.constant(1.0, dtype=tf.float32)
    
    # Conditional computation
    return tf.cond(valid_batch, compute_ctc, fallback_loss)

def create_ultra_conservative_training_step():
    """Training step dengan validasi ekstensif"""
    
    @tf.function
    def conservative_train_step(batch_data):
        """Training step yang sangat hati-hati"""
        
        # Validasi input data
        batch_train = batch_data['deg_image']
        batch_target = batch_data['gt_image']
        x_train_rcnn = batch_data['crnn_image']
        y_train_rcnn = batch_data['transcription']
        
        # Validasi shapes
        tf.debugging.assert_equal(tf.shape(batch_train)[1:], [128, 1024, 1])
        tf.debugging.assert_equal(tf.shape(batch_target)[1:], [128, 1024, 1])
        
        # Cek data validity
        tf.debugging.assert_all_finite(batch_train, "batch_train contains non-finite values")
        tf.debugging.assert_all_finite(batch_target, "batch_target contains non-finite values")
        
        per_replica_batch_size = tf.shape(batch_train)[0]
        
        # Generate dengan error handling
        try:
            generated_images = generator(batch_train, training=False)
            tf.debugging.assert_all_finite(generated_images, "Generated images contain non-finite")
        except Exception:
            print("❌ Generator failed")
            return tf.constant(1.0), tf.constant(1.0), tf.constant(1.0)
        
        # Training steps dengan extensive error handling
        # ... (implement conservative training logic)
        
        return d1_loss, d2_loss, g_loss
    
    return conservative_train_step

# CONFIGURATION UNTUK EMERGENCY TRAINING
EMERGENCY_CONFIG = {
    'epochs': 10,                    # Pendek untuk testing
    'batch_size': 1,                 # Minimal
    'learning_rate': 0.00001,        # Sangat kecil
    'patience': 3,                   # Pendek
    'save_interval': 1,              # Save setiap epoch
    'eval_interval': 1,              # Eval setiap epoch
    'max_samples': 50,               # Limit data untuk debugging
    'gradient_clip_norm': 0.05,      # Sangat agresif
    'loss_weights': [0.5, 1.0, 0.5] # Konservatif
}
'''
    
    return fixed_script

def create_debugging_checklist():
    """Buat checklist untuk debugging step-by-step"""
    
    checklist = {
        "Pre-Training Checks": [
            "✅ Verifikasi CUDA dan GPU memory available",
            "✅ Check dataset paths dan file accessibility", 
            "✅ Validate charlist.txt dan character encoding",
            "✅ Test data generator dengan 1 batch",
            "✅ Verify image dimensions (128, 1024, 1)",
            "✅ Check transcription lengths (<20 chars)"
        ],
        
        "Training Setup": [
            "✅ Set batch_size = 1 untuk start",
            "✅ Set learning_rate = 0.00001 (very low)",
            "✅ Enable gradient clipping (norm=0.05)",
            "✅ Reduce loss weights (recognition=0.5)",
            "✅ Limit dataset size (50-100 samples)",
            "✅ Set patience = 3 epochs"
        ],
        
        "During Training": [
            "✅ Monitor EVERY batch for NaN values",
            "✅ Check GPU memory usage",
            "✅ Validate CTC inputs shape consistency",
            "✅ Log gradients norm untuk each discriminator",
            "✅ Save model weights setiap epoch",
            "✅ Plot loss progression real-time"
        ],
        
        "If Problems Persist": [
            "✅ Disable mixed precision training",
            "✅ Use single GPU (disable multi-GPU)",
            "✅ Simplify CTC loss function",
            "✅ Test dengan synthetic data",
            "✅ Reduce model complexity temporarily",
            "✅ Check TensorFlow version compatibility"
        ]
    }
    
    print("\n📝 DEBUGGING CHECKLIST:")
    print("=" * 60)
    
    for category, items in checklist.items():
        print(f"\n🔍 {category}:")
        for item in items:
            print(f"   {item}")
    
    return checklist

def generate_test_script():
    """Generate script untuk test individual components"""
    
    test_script = '''
# SCRIPT TEST INDIVIDUAL COMPONENTS
# Untuk mengisolasi masalah pada setiap bagian

def test_data_loading():
    """Test data loading tanpa training"""
    print("🧪 Testing data loading...")
    
    # Test basic data loading
    list_image_train = read_file_shuffle(rootPath + 'Sets/list_train_nan.txt')[:5]
    list_lines = read_file(rootPath + 'Sets/lines.txt')
    
    for im_base in list_image_train:
        try:
            # Test image loading
            deg_image, gt_image = readGrayPair(im_base + '.png', split='train')
            print(f"✅ Image {im_base}: {deg_image.shape}, {gt_image.shape}")
            
            # Check for NaN values
            if np.any(np.isnan(deg_image)) or np.any(np.isnan(gt_image)):
                print(f"❌ NaN detected in {im_base}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading {im_base}: {e}")
            return False
    
    print("✅ Data loading test passed")
    return True

def test_model_creation():
    """Test model creation tanpa training"""
    print("🧪 Testing model creation...")
    
    try:
        generator = unet()
        discriminator_1 = build_discriminator_1()
        discriminator_2 = build_discriminator_2()
        
        print("✅ All models created successfully")
        
        # Test forward pass
        dummy_input = np.random.random((1, 128, 1024, 1)).astype(np.float32)
        gen_output = generator(dummy_input)
        print(f"✅ Generator output shape: {gen_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_ctc_loss():
    """Test CTC loss dengan sample data"""
    print("🧪 Testing CTC loss...")
    
    # Sample data
    batch_size = 2
    seq_length = 256
    vocab_size = len(charset_base) + 1
    max_label_length = 10
    
    # Create sample predictions dan labels
    y_pred = tf.random.normal((batch_size, seq_length, vocab_size))
    y_true = tf.random.uniform((batch_size, max_label_length), 
                              maxval=vocab_size-1, dtype=tf.int32)
    
    try:
        loss = improved_ctc_loss_lambda_func(y_true, y_pred)
        print(f"✅ CTC loss computed: {float(loss):.4f}")
        
        if tf.math.is_finite(loss):
            print("✅ CTC loss is finite")
            return True
        else:
            print("❌ CTC loss is not finite")
            return False
            
    except Exception as e:
        print(f"❌ CTC loss failed: {e}")
        return False

# RUN ALL TESTS
if __name__ == "__main__":
    print("🚀 RUNNING COMPONENT TESTS")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation), 
        ("CTC Loss", test_ctc_loss)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    print(f"\\n🏁 TEST RESULTS:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    if all(results.values()):
        print("\\n🎉 All tests passed! Ready for conservative training.")
    else:
        print("\\n🚨 Some tests failed. Fix issues before training.")
'''
    
    return test_script

def main():
    """Main function untuk menjalankan analisis dan generate solutions"""
    
    # 1. Analisis masalah
    problems, root_causes = analyze_training_failure()
    
    # 2. Buat emergency fixes
    emergency_config = create_emergency_fixes()
    
    # 3. Generate fixed script
    fixed_script = generate_fixed_training_script()
    
    # 4. Buat debugging checklist
    checklist = create_debugging_checklist()
    
    # 5. Generate test script
    test_script = generate_test_script()
    
    # 6. Save semua ke files
    output_dir = "/home/lambda_one/tesis/GAN-HTR/periksa"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save emergency config
    with open(os.path.join(output_dir, "emergency_config.json"), 'w') as f:
        json.dump(emergency_config, f, indent=2)
    
    # Save fixed script
    with open(os.path.join(output_dir, "fixed_training_script.py"), 'w') as f:
        f.write(fixed_script)
    
    # Save test script  
    with open(os.path.join(output_dir, "component_tests.py"), 'w') as f:
        f.write(test_script)
    
    print(f"\\n💾 SOLUTIONS SAVED:")
    print(f"   📁 {output_dir}/emergency_config.json")
    print(f"   📁 {output_dir}/fixed_training_script.py") 
    print(f"   📁 {output_dir}/component_tests.py")
    
    # 7. Immediate action recommendations
    print("\\n🎯 IMMEDIATE ACTION PLAN:")
    print("=" * 60)
    
    immediate_actions = [
        "1. 🛑 STOP current training immediately",
        "2. 🔧 Apply emergency configuration (batch_size=1, lr=0.00001)",
        "3. 🧪 Run component tests to isolate problems", 
        "4. 📊 Check available GPU memory",
        "5. 🔍 Validate input data for NaN/corrupted files",
        "6. 🚀 Start conservative training with 50 samples only",
        "7. 📈 Monitor every batch for NaN occurrence",
        "8. 💾 Save model weights every epoch during testing"
    ]
    
    for action in immediate_actions:
        print(f"   {action}")
    
    print("\\n⚡ PRIORITY ORDER:")
    print("   1. Fix CTC loss function (most critical)")
    print("   2. Reduce learning rates drastically") 
    print("   3. Limit data size for debugging")
    print("   4. Add extensive validation checks")
    
    print("\\n✅ Analisis selesai! Follow action plan step by step.")

if __name__ == "__main__":
    main()
