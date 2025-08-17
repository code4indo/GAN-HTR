#!/usr/bin/env python3
"""
Test script untuk memverifikasi bahwa perbaikan LeakyReLU berhasil
"""

import warnings
import sys

def test_leaky_relu_import():
    """
    Test import LeakyReLU dan penggunaan parameter negative_slope
    """
    print("🧪 Testing LeakyReLU import dan parameter...")
    
    try:
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Import dan test LeakyReLU
            from tensorflow.keras.layers import LeakyReLU
            
            # Test dengan parameter baru
            layer = LeakyReLU(negative_slope=0.2)
            print(f"✅ LeakyReLU dengan negative_slope berhasil dibuat: {layer}")
            
            # Check for any deprecation warnings
            deprecation_warnings = [warning for warning in w if 'deprecated' in str(warning.message).lower()]
            
            if deprecation_warnings:
                print("⚠️  Masih ada deprecation warnings:")
                for warning in deprecation_warnings:
                    print(f"   - {warning.message}")
                return False
            else:
                print("✅ Tidak ada deprecation warnings!")
                return True
                
    except Exception as e:
        print(f"❌ Error saat testing: {e}")
        return False

def test_model_creation():
    """
    Test pembuatan model sederhana dengan LeakyReLU
    """
    print("\n🧪 Testing model creation dengan LeakyReLU...")
    
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, LeakyReLU
        from tensorflow.keras.models import Sequential
        
        # Suppress TensorFlow info messages
        tf.get_logger().setLevel('ERROR')
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Create simple model
            model = Sequential([
                Dense(64)),
                LeakyReLU(negative_slope=0.2),
                Dense(32),
                LeakyReLU(negative_slope=0.1),
                Dense(1, activation='sigmoid')
            ])
            
            print(f"✅ Model berhasil dibuat dengan {len(model.layers)} layers")
            
            # Check for deprecation warnings
            deprecation_warnings = [warning for warning in w if 'deprecated' in str(warning.message).lower()]
            
            if deprecation_warnings:
                print("⚠️  Ada deprecation warnings saat membuat model:")
                for warning in deprecation_warnings:
                    print(f"   - {warning.message}")
                return False
            else:
                print("✅ Model dibuat tanpa deprecation warnings!")
                return True
                
    except Exception as e:
        print(f"❌ Error saat testing model creation: {e}")
        return False

def main():
    """
    Main test function
    """
    print("🔍 Verifikasi perbaikan LeakyReLU deprecation warning")
    print("=" * 60)
    
    test1_passed = test_leaky_relu_import()
    test2_passed = test_model_creation()
    
    print("\n" + "=" * 60)
    print("📊 Hasil Test:")
    print(f"   ✅ Import test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"   ✅ Model creation test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 Semua test PASSED! Perbaikan berhasil!")
        print("💡 Tips:")
        print("   - Parameter 'alpha' sudah diganti dengan 'negative_slope'")
        print("   - Kode sekarang kompatibel dengan Keras 3.x")
        print("   - Tidak ada lagi deprecation warnings!")
        return True
    else:
        print("\n❌ Ada test yang FAILED. Periksa kembali kode Anda.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
