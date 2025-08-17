#!/usr/bin/env python3
"""
Script untuk validasi database sebelum menjalankan fungsi PSNR
"""

import os
from PIL import Image

def validate_database():
    """Validasi apakah database sudah sesuai format"""
    
    print("🔍 VALIDASI DATABASE UNTUK FUNGSI PSNR")
    print("=" * 50)
    
    # Cek struktur direktori
    required_dirs = [
        "datasets/nan_raw_biner/test/images",
        "datasets/nan_raw_biner/train/images", 
        "datasets/nan_raw_biner/validation/images",
        "Sets"
    ]
    
    print("\n📁 Memeriksa struktur direktori...")
    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            print(f"✅ {dir_path} ({file_count} files)")
        else:
            print(f"❌ {dir_path} - TIDAK ADA!")
            missing_dirs.append(dir_path)
    
    # Cek file lists
    print("\n📄 Memeriksa file lists...")
    list_files = [
        "Sets/list_test_nan.txt",
        "Sets/list_train_nan.txt", 
        "Sets/list_valid_nan.txt"
    ]
    
    missing_lists = []
    for list_file in list_files:
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                count = len([line for line in f.readlines() if line.strip()])
            print(f"✅ {list_file} - {count} entries")
        else:
            print(f"❌ {list_file} - TIDAK ADA!")
            missing_lists.append(list_file)
    
    # Cek konsistensi file (test set)
    print("\n🔗 Memeriksa konsistensi file...")
    test_list_file = "Sets/list_test_nan.txt"
    test_img_dir = "datasets/nan_raw_biner/test/images"
    
    if os.path.exists(test_list_file) and os.path.exists(test_img_dir):
        with open(test_list_file, 'r') as f:
            listed_files = [line.strip() for line in f.readlines() if line.strip()]
        
        if not listed_files:
            print("❌ File list kosong!")
        else:
            # Cek 5 file pertama
            check_count = min(5, len(listed_files))
            missing_files = []
            existing_files = []
            
            for base_name in listed_files[:check_count]:
                # Cek apakah file ada (jpg atau png)
                jpg_path = os.path.join(test_img_dir, base_name + '.jpg')
                png_path = os.path.join(test_img_dir, base_name + '.png')
                
                if os.path.exists(jpg_path):
                    existing_files.append((base_name, 'jpg'))
                elif os.path.exists(png_path):
                    existing_files.append((base_name, 'png'))
                else:
                    missing_files.append(base_name)
            
            if missing_files:
                print(f"❌ File hilang: {len(missing_files)} dari {check_count} yang dicek")
                for missing in missing_files:
                    print(f"   - {missing}")
            else:
                print(f"✅ File konsisten: {len(existing_files)} dari {check_count} ditemukan")
                # Tampilkan contoh format
                if existing_files:
                    example = existing_files[0]
                    print(f"   Contoh: {example[0]}.{example[1]}")
    
    # Cek direktori hasil training
    print("\n🎯 Memeriksa direktori hasil training...")
    result_patterns = [
        "ResultGanS_iam_OP_debug",
        "ResultS_iam_OP_debug", 
        "Result_iam_OP_debug"
    ]
    
    found_result_dirs = []
    for pattern in result_patterns:
        if os.path.exists(pattern):
            epochs = [d for d in os.listdir(pattern) if d.startswith('epoch') and os.path.isdir(os.path.join(pattern, d))]
            if epochs:
                latest_epoch = max(epochs, key=lambda x: int(x.replace('epoch', '')))
                epoch_path = os.path.join(pattern, latest_epoch)
                file_count = len([f for f in os.listdir(epoch_path) if f.endswith('.png')])
                print(f"✅ {pattern}/{latest_epoch} ({file_count} enhanced images)")
                found_result_dirs.append((pattern, latest_epoch, file_count))
            else:
                print(f"⚠️  {pattern} ada tapi tidak ada epoch directories")
    
    if not found_result_dirs:
        print("❌ Tidak ada direktori hasil training ditemukan")
        print("   Jalankan training terlebih dahulu untuk menghasilkan enhanced images")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 RINGKASAN VALIDASI")
    
    if missing_dirs:
        print(f"❌ Direktori hilang: {len(missing_dirs)}")
    else:
        print("✅ Semua direktori tersedia")
    
    if missing_lists:
        print(f"❌ File lists hilang: {len(missing_lists)}")
    else:
        print("✅ Semua file lists tersedia")
    
    if found_result_dirs:
        print(f"✅ Direktori hasil training: {len(found_result_dirs)} ditemukan")
    else:
        print("❌ Tidak ada hasil training")
    
    # Rekomendasi
    print("\n💡 REKOMENDASI:")
    if missing_dirs:
        print("1. Buat struktur direktori yang hilang")
        print("2. Copy gambar ke direktori yang sesuai")
    
    if missing_lists:
        print("3. Jalankan: poetry run python periksa/create_file_lists.py")
    
    if not found_result_dirs:
        print("4. Jalankan training untuk menghasilkan enhanced images")
        print("   poetry run python jnm_GAN_AHTR.py --scenario S_iam_OP_debug")
    
    if not (missing_dirs or missing_lists) and found_result_dirs:
        print("🎉 Database siap untuk menjalankan fungsi PSNR!")
        print("   poetry run python periksa/fixed_get_psnr_nan.py")

if __name__ == '__main__':
    validate_database()
