#!/usr/bin/env python3
"""
Script untuk membuat file lists dari gambar yang ada
"""

import os
from glob import glob

def create_file_lists():
    """Membuat file lists dari gambar yang ada"""
    
    # Path ke direktori images
    test_dir = "datasets/nan_raw_biner/test/images"
    train_dir = "datasets/nan_raw_biner/train/images"
    valid_dir = "datasets/nan_raw_biner/validation/images"
    
    # Fungsi untuk extract base name tanpa ekstensi
    def get_base_names(directory):
        if not os.path.exists(directory):
            print(f"⚠️  Direktori tidak ada: {directory}")
            return []
        
        files = glob(os.path.join(directory, "*.*"))
        base_names = []
        
        for file_path in files:
            filename = os.path.basename(file_path)
            # Hapus ekstensi (.jpg, .png, dll)
            base_name = os.path.splitext(filename)[0]
            base_names.append(base_name)
        
        return sorted(base_names)
    
    # Buat file lists
    datasets = {
        'test': (test_dir, 'Sets/list_test_nan.txt'),
        'train': (train_dir, 'Sets/list_train_nan.txt'),
        'validation': (valid_dir, 'Sets/list_valid_nan.txt')
    }
    
    print("🔧 Membuat file lists dari gambar yang ada...")
    
    for dataset_name, (source_dir, output_file) in datasets.items():
        base_names = get_base_names(source_dir)
        
        if not base_names:
            print(f"❌ {dataset_name}: Tidak ada gambar di {source_dir}")
            continue
        
        # Tulis ke file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for base_name in base_names:
                f.write(base_name + '\n')
        
        print(f"✅ {dataset_name}: {len(base_names)} files -> {output_file}")
    
    print("\n📄 File lists berhasil dibuat!")

if __name__ == '__main__':
    create_file_lists()
