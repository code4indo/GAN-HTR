#!/usr/bin/env python3
"""
Script untuk mengatasi deprecation warning LeakyReLU alpha parameter.
Mengganti semua 'alpha' menjadi 'negative_slope' dalam LeakyReLU layers.
"""

import os
import re
import glob

def fix_leaky_relu_in_file(file_path):
    """
    Mengganti parameter alpha menjadi negative_slope dalam file Python
    """
    print(f"Memeriksa file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern untuk mencari LeakyReLU dengan parameter alpha
        pattern = r'LeakyReLU\s*\(\s*alpha\s*='
        replacement = r'LeakyReLU(negative_slope='
        
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

def main():
    """
    Main function untuk memperbaiki semua file Python yang menggunakan LeakyReLU
    """
    print("🔧 Memperbaiki deprecation warning LeakyReLU alpha parameter...")
    print("=" * 60)
    
    # Daftar file yang perlu diperbaiki berdasarkan hasil grep
    files_to_fix = [
        "/home/lambda_one/tesis/GAN-HTR/jnm_GAN_AHTR.py",
        "/home/lambda_one/tesis/GAN-HTR/jnm_GAN_AHTR copy.py",
        "/home/lambda_one/tesis/GAN-HTR/periksa/test_enhancement_existing_model.py",
        "/home/lambda_one/tesis/GAN-HTR/periksa/quick_training_test.py",
        "/home/lambda_one/tesis/GAN-HTR/periksa/train_fixed_model.py",
        "/home/lambda_one/tesis/GAN-HTR/dibco_TL_2010.py",
        "/home/lambda_one/tesis/GAN-HTR/GAN_AHTR.py"
    ]
    
    fixed_count = 0
    total_count = len(files_to_fix)
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_leaky_relu_in_file(file_path):
                fixed_count += 1
        else:
            print(f"⚠️  File tidak ditemukan: {file_path}")
    
    print("=" * 60)
    print(f"✅ Selesai! {fixed_count}/{total_count} file diperbaiki.")
    print("\n📋 Ringkasan perubahan:")
    print("   - Parameter 'alpha' → 'negative_slope' di LeakyReLU layers")
    print("   - Mengatasi deprecation warning di Keras 3.x")
    print("\n🚀 Sekarang Anda dapat menjalankan kode tanpa warning!")

if __name__ == "__main__":
    main()
