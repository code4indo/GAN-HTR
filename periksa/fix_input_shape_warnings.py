#!/usr/bin/env python3
"""
Script untuk mengatasi Keras warning tentang input_shape pada layer Reshape.
Warning: Do not pass an `input_shape`/`input_dim` argument to a layer.

Masalah ini terjadi karena:
1. Keras 3.x menghapus support untuk input_shape pada layer non-input
2. Layer Reshape dengan input_shape menyebabkan warning
3. Sequential models harus menggunakan Input() layer sebagai layer pertama
"""

import os
import re
import glob

def fix_reshape_input_shape_in_file(file_path):
    """
    Menghapus parameter input_shape dari layer Reshape
    """
    print(f"Memeriksa file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern untuk mencari Reshape dengan input_shape
        # Contoh: Reshape((1024,128,1 ), input_shape=(128,1024,1))
        pattern = r'Reshape\s*\(\s*([^,)]+)\s*,\s*input_shape\s*=\s*[^)]+\)'
        replacement = r'Reshape(\1)'
        
        # Ganti semua occurrence
        content = re.sub(pattern, replacement, content)
        
        # Jika ada perubahan, simpan file
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ File diperbaiki: {file_path}")
            return True
        else:
            print(f"ℹ️  Tidak ada perubahan diperlukan: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error saat memproses {file_path}: {e}")
        return False

def fix_sequential_input_shape_usage(file_path):
    """
    Memperbaiki penggunaan input_shape pada layer non-input dalam Sequential model
    """
    print(f"Memeriksa Sequential model di: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = False
        
        # Pattern untuk Dense dengan input_shape (bukan layer pertama)
        # Contoh: Dense(64, input_shape=(100,))
        if 'Dense(' in content and 'input_shape=' in content:
            # Cari pattern yang bermasalah
            pattern = r'Dense\s*\(\s*(\d+)\s*,\s*input_shape\s*=\s*([^)]+)\)'
            matches = re.findall(pattern, content)
            
            if matches:
                print(f"   Ditemukan {len(matches)} Dense layer dengan input_shape")
                # Untuk Sequential model, layer pertama harus menggunakan Input()
                # Layer lainnya tidak boleh menggunakan input_shape
                
                # Ganti Dense dengan input_shape menjadi Dense biasa
                content = re.sub(pattern, r'Dense(\1)', content)
                changes_made = True
        
        # Jika ada perubahan, simpan file
        if changes_made and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Sequential model diperbaiki: {file_path}")
            return True
        else:
            print(f"ℹ️  Sequential model tidak memerlukan perbaikan: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error saat memproses Sequential model {file_path}: {e}")
        return False

def check_keras_version():
    """
    Check versi Keras untuk informasi
    """
    try:
        import tensorflow as tf
        print(f"🔍 TensorFlow version: {tf.__version__}")
        
        import keras
        print(f"🔍 Keras version: {keras.__version__}")
        
        return True
    except Exception as e:
        print(f"❌ Error checking versions: {e}")
        return False

def main():
    """
    Main function untuk memperbaiki input_shape warnings
    """
    print("🔧 Memperbaiki Keras input_shape deprecation warnings...")
    print("=" * 65)
    
    # Check Keras version
    check_keras_version()
    print()
    
    # Daftar file yang perlu diperbaiki berdasarkan hasil grep
    files_to_fix = [
        "/home/lambda_one/tesis/GAN-HTR/jnm_GAN_AHTR.py",
        "/home/lambda_one/tesis/GAN-HTR/jnm_GAN_AHTR copy.py",
        "/home/lambda_one/tesis/GAN-HTR/create_working_file.py",
        "/home/lambda_one/tesis/GAN-HTR/GAN_AHTR.py",
        "/home/lambda_one/tesis/GAN-HTR/periksa/train_gan_nan.py",
        "/home/lambda_one/tesis/GAN-HTR/periksa/train_gan_optimized.py",
        "/home/lambda_one/tesis/GAN-HTR/periksa/test_leaky_relu_fix.py",
        "/home/lambda_one/tesis/GAN-HTR/periksa/benchmark_hardware.py"
    ]
    
    fixed_count = 0
    total_count = len(files_to_fix)
    
    print("🔧 Memperbaiki Reshape input_shape...")
    print("-" * 40)
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_reshape_input_shape_in_file(file_path):
                fixed_count += 1
        else:
            print(f"⚠️  File tidak ditemukan: {file_path}")
    
    print("\n🔧 Memperbaiki Sequential model input_shape...")
    print("-" * 40)
    
    sequential_fixed = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_sequential_input_shape_usage(file_path):
                sequential_fixed += 1
    
    print("=" * 65)
    print(f"✅ Selesai! {fixed_count}/{total_count} file Reshape diperbaiki.")
    print(f"✅ {sequential_fixed} file Sequential model diperbaiki.")
    print("\n📋 Ringkasan perubahan:")
    print("   - Menghapus 'input_shape' dari layer Reshape")
    print("   - Memperbaiki penggunaan input_shape di Sequential models")
    print("   - Mengatasi deprecation warning di Keras 3.x")
    print("\n💡 Untuk Sequential models, gunakan Input() layer sebagai layer pertama:")
    print("   model = Sequential([")
    print("       Input(shape=(100,)),  # Gunakan Input() layer")
    print("       Dense(64),            # Tidak perlu input_shape")
    print("       Dense(32)")
    print("   ])")
    print("\n🚀 Sekarang Anda dapat menjalankan kode tanpa input_shape warnings!")

if __name__ == "__main__":
    main()
