# Format Database untuk Fungsi PSNR

## Struktur Database yang Diharapkan

Berdasarkan analisis fungsi `get_psnr_nan()`, berikut adalah format database yang diharapkan untuk dapat menggunakan fungsi perhitungan PSNR:

### 1. Struktur Direktori Ground Truth Images

```
datasets/
└── nan_raw_biner/
    ├── train/
    │   └── images/
    │       ├── 000_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1210.jpg
    │       ├── 001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg
    │       └── ...
    ├── test/
    │   └── images/
    │       ├── 000_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1210.jpg
    │       ├── 001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg
    │       └── ...
    └── validation/
        └── images/
            ├── 000_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1210.jpg
            ├── 001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg
            └── ...
```

### 2. Struktur Direktori Enhanced Images (Hasil GAN)

```
Result[SCENARIO]/
├── epoch0/
│   ├── 000_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1210.jpg.png
│   ├── 001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg.png
│   └── ...
├── epoch1/
│   ├── 000_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1210.jpg.png
│   ├── 001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg.png
│   └── ...
└── epochN/
    ├── 000_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1210.jpg.png
    ├── 001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg.png
    └── ...
```

### 3. File Lists

```
Sets/
├── list_test_nan.txt     # Daftar nama file untuk testing (tanpa ekstensi)
├── list_train_nan.txt    # Daftar nama file untuk training (tanpa ekstensi)
└── list_valid_nan.txt    # Daftar nama file untuk validation (tanpa ekstensi)
```

## Contoh Implementasi

### Contoh Isi File `Sets/list_test_nan.txt`:
```
000_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1210
001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211
002_NL-HaNA_1.04.02_8740_0147.tif_r1l1
003_NL-HaNA_1.04.02_8740_0147.tif_r1l10
004_NL-HaNA_1.04.02_8740_0147.tif_r1l11
```

### Contoh Nama File Ground Truth:
```
datasets/nan_raw_biner/test/images/000_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1210.jpg
```

### Contoh Nama File Enhanced (hasil GAN):
```
ResultS_iam_OP_debug/epoch2/000_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1210.jpg.png
```

## Persyaratan Format

### 1. Nama File
- **Ground Truth**: `[base_name].[jpg|png]`
- **Enhanced**: `[base_name].jpg.png` atau `[base_name].png`
- **List File**: hanya `[base_name]` (tanpa ekstensi)

### 2. Format Gambar
- **Ground Truth**: JPG atau PNG
- **Enhanced**: PNG (hasil dari GAN)
- **Resolusi**: Akan di-resize otomatis ke 1024x128 pixels
- **Color Mode**: Akan dikonversi ke grayscale

### 3. Struktur Direktori
- **Ground Truth**: `datasets/nan_raw_biner/test/images/`
- **Enhanced**: `Result[SCENARIO]/epoch[N]/`
- **Lists**: `Sets/`

## Script untuk Mempersiapkan Database

### 1. Script untuk Membuat Structure Direktori:

```bash
#!/bin/bash
# create_database_structure.sh

# Buat struktur direktori
mkdir -p datasets/nan_raw_biner/{train,test,validation}/images
mkdir -p Sets

# Buat direktori hasil (contoh)
mkdir -p ResultS_iam_OP_debug/{epoch0,epoch1,epoch2,epoch3}

echo "✅ Struktur direktori berhasil dibuat!"
```

### 2. Script untuk Membuat File Lists:

```python
#!/usr/bin/env python3
# create_file_lists.py

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
    
    for dataset_name, (source_dir, output_file) in datasets.items():
        base_names = get_base_names(source_dir)
        
        # Tulis ke file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for base_name in base_names:
                f.write(base_name + '\n')
        
        print(f"✅ {dataset_name}: {len(base_names)} files -> {output_file}")

if __name__ == '__main__':
    create_file_lists()
```

### 3. Script Validasi Database:

```python
#!/usr/bin/env python3
# validate_database.py

import os
from PIL import Image

def validate_database():
    """Validasi apakah database sudah sesuai format"""
    
    # Cek struktur direktori
    required_dirs = [
        "datasets/nan_raw_biner/test/images",
        "datasets/nan_raw_biner/train/images", 
        "datasets/nan_raw_biner/validation/images",
        "Sets"
    ]
    
    print("🔍 Memeriksa struktur direktori...")
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - TIDAK ADA!")
    
    # Cek file lists
    print("\n🔍 Memeriksa file lists...")
    list_files = [
        "Sets/list_test_nan.txt",
        "Sets/list_train_nan.txt", 
        "Sets/list_valid_nan.txt"
    ]
    
    for list_file in list_files:
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                count = len(f.readlines())
            print(f"✅ {list_file} - {count} entries")
        else:
            print(f"❌ {list_file} - TIDAK ADA!")
    
    # Cek konsistensi file
    print("\n🔍 Memeriksa konsistensi file...")
    test_list_file = "Sets/list_test_nan.txt"
    test_img_dir = "datasets/nan_raw_biner/test/images"
    
    if os.path.exists(test_list_file) and os.path.exists(test_img_dir):
        with open(test_list_file, 'r') as f:
            listed_files = [line.strip() for line in f.readlines() if line.strip()]
        
        missing_files = []
        for base_name in listed_files[:10]:  # Cek 10 file pertama
            # Cek apakah file ada (jpg atau png)
            jpg_path = os.path.join(test_img_dir, base_name + '.jpg')
            png_path = os.path.join(test_img_dir, base_name + '.png')
            
            if not (os.path.exists(jpg_path) or os.path.exists(png_path)):
                missing_files.append(base_name)
        
        if missing_files:
            print(f"❌ File hilang: {len(missing_files)} dari {len(listed_files[:10])} yang dicek")
            for missing in missing_files[:3]:
                print(f"   - {missing}")
        else:
            print(f"✅ File konsisten: 10 file pertama ditemukan")

if __name__ == '__main__':
    validate_database()
```

## Cara Penggunaan

### 1. Persiapan Database Baru:

```bash
# 1. Buat struktur direktori
bash create_database_structure.sh

# 2. Copy file gambar Anda ke direktori yang sesuai
cp /path/to/your/images/* datasets/nan_raw_biner/test/images/

# 3. Buat file lists
poetry run python create_file_lists.py

# 4. Validasi database
poetry run python validate_database.py
```

### 2. Menjalankan Fungsi PSNR:

```bash
# Setelah training selesai dan ada hasil di ResultS_iam_OP_debug/
poetry run python periksa/fixed_get_psnr_nan.py
```

### 3. Menggunakan dalam Script:

```python
from periksa.fixed_get_psnr_nan import get_psnr_nan

# Hitung PSNR untuk scenario tertentu
psnr_result = get_psnr_nan('S_iam_OP_debug', epoch_num=2)
print(f"Average PSNR: {psnr_result:.2f} dB")
```

## Catatan Penting

1. **Nama File Harus Konsisten**: Base name di file list harus sama dengan nama file gambar (tanpa ekstensi)
2. **Format Ekstensi**: Ground truth bisa .jpg atau .png, enhanced biasanya .png
3. **Resolusi Otomatis**: Semua gambar akan di-resize ke 1024x128 pixels
4. **Path Absolut**: Pastikan menjalankan dari root directory project
5. **Scenario Name**: Harus sesuai dengan nama direktori hasil (Result[SCENARIO])

## Troubleshooting

### Jika File Tidak Ditemukan:
1. Periksa nama file di list vs nama file aktual
2. Periksa ekstensi file (.jpg vs .png)
3. Periksa case sensitivity (Linux sensitive)
4. Pastikan path relatif dari root project

### Jika PSNR Tidak Masuk Akal:
1. Periksa format gambar (grayscale vs color)
2. Periksa resolusi gambar
3. Periksa apakah enhanced image benar-benar hasil enhancement
