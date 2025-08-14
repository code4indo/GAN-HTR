# 📖 Manual Penggunaan GAN-HTR Document Enhancement

## 🚀 Panduan Lengkap Memperbaiki Dokumen Rusak

### 📋 Persyaratan Sistem
- Python 3.10+
- TensorFlow 2.16+
- OpenCV (cv2)
- Model GAN-HTR yang sudah ditraining
- GPU NVIDIA (recommended untuk kecepatan)

### 🛠️ Instalasi
```bash
# Clone repository
git clone https://github.com/code4indo/GAN-HTR.git
cd GAN-HTR

# Install dependencies
pip install -r requirements.txt
# atau
poetry install
```

## 🎯 Cara Memperbaiki Dokumen Rusak

### 1. **Metode Simple (Recommended untuk pemula)**

```bash
# Perbaiki dokumen tunggal
python simple_enhancement_test.py
```

**Cara kerja:**
- Script akan otomatis mencari model terbaru
- Memproses semua dokumen yang ada di direktori (a.png, b.jpg, dll)
- Menghasilkan file enhanced dan comparison

### 2. **Metode CLI (Untuk penggunaan advanced)**

```bash
# Sintaks dasar
python full_document_enhancement.py \
    --input [FILE_INPUT] \
    --output [FILE_OUTPUT] \
    --model [PATH_MODEL] \
    --method [line_detection|sliding_window]

# Contoh penggunaan
python full_document_enhancement.py \
    --input dokumen_rusak.jpg \
    --output dokumen_diperbaiki.png \
    --model ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5 \
    --method sliding_window

# Dengan menyimpan proses intermediate
python full_document_enhancement.py \
    --input dokumen_rusak.jpg \
    --output dokumen_diperbaiki.png \
    --model ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5 \
    --method sliding_window \
    --save-intermediates
```

### 3. **Metode Python Script (Untuk integrasi)**

```python
from simple_enhancement_test import (
    build_unet_generator, 
    simple_preprocess_document,
    simple_enhance_segments,
    simple_reconstruct_document,
    create_simple_comparison
)
import cv2

# Load model
generator = build_unet_generator()
generator.load_weights('./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5')

# Proses dokumen
doc_path = 'dokumen_rusak.jpg'
segments, coordinates = simple_preprocess_document(doc_path)
enhanced_segments = simple_enhance_segments(generator, segments)

# Rekonstruksi
original_img = cv2.imread(doc_path, cv2.IMREAD_GRAYSCALE)
reconstructed = simple_reconstruct_document(enhanced_segments, coordinates, original_img.shape)

# Simpan hasil
cv2.imwrite('dokumen_diperbaiki.png', reconstructed)
```

## 📁 Input dan Output

### 📥 **Format Input yang Didukung:**
- `.jpg`, `.jpeg` - Foto dokumen
- `.png` - Scan dokumen
- `.tiff` - Dokumen berkualitas tinggi
- Ukuran: Apapun (otomatis disesuaikan)

### 📤 **Output yang Dihasilkan:**
- **File Enhanced:** Dokumen yang sudah diperbaiki
- **File Comparison:** Perbandingan sebelum vs sesudah
- **Intermediate Files** (opsional): Segmen-segmen proses

## ⚙️ Parameter dan Konfigurasi

### 🔧 **Parameter CLI:**
- `--input`: Path file dokumen yang akan diperbaiki
- `--output`: Path output dokumen hasil perbaikan
- `--model`: Path model weights (.h5)
- `--method`: Metode preprocessing (`line_detection` atau `sliding_window`)
- `--save-intermediates`: Simpan file proses intermediate

### 🎛️ **Konfigurasi Advanced:**
```python
# Dalam script preprocessing
segment_height = 128      # Tinggi segment (jangan diubah)
segment_width = 1024      # Lebar segment (jangan diubah)
overlap = 0.1             # Overlap antar segment (0.1 = 10%)
```

## 🎯 Contoh Kasus Penggunaan

### 📄 **Case 1: Dokumen Scan Berkualitas Rendah**
```bash
python simple_enhancement_test.py
# File a.png dan b.jpg akan otomatis diproses
```

### 📊 **Case 2: Dokumen Besar dengan Multiple Pages**
```bash
python full_document_enhancement.py \
    --input dokumen_besar.tiff \
    --output hasil_perbaikan.png \
    --model ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5 \
    --method sliding_window \
    --save-intermediates
```

### 🔄 **Case 3: Batch Processing Multiple Files**
```bash
# Buat script bash untuk multiple files
for file in *.jpg; do
    python full_document_enhancement.py \
        --input "$file" \
        --output "enhanced_$file" \
        --model ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5
done
```

## 🛠️ Troubleshooting

### ❌ **Error: Model not found**
```bash
# Pastikan path model benar
ls -la ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5

# Atau gunakan model alternatif
ls -la ./ResultGanS_S_nan_OP_SIMPLE/epoch_*/weights/generator.weights.h5
```

### ❌ **Error: CUDA out of memory**
```python
# Kurangi ukuran batch dalam script
# Edit di simple_enhancement_test.py:
# Proses satu per satu jika memory terbatas
```

### ❌ **Error: No segments generated**
```bash
# Coba method yang berbeda
python full_document_enhancement.py \
    --method sliding_window  # Jika line_detection gagal
```

## 📊 Tips Optimasi

### 🚀 **Untuk Kecepatan:**
- Gunakan GPU jika tersedia
- Proses dokumen dalam batch kecil
- Gunakan format PNG untuk output berkualitas tinggi

### 🎨 **Untuk Kualitas:**
- Gunakan method `sliding_window` untuk dokumen kompleks
- Aktifkan `--save-intermediates` untuk debugging
- Bandingkan hasil dengan file comparison

### 💾 **Untuk Memory:**
- Proses dokumen besar secara bertahap
- Tutup aplikasi lain saat processing
- Monitor penggunaan VRAM

## 📈 Expected Results

### ✅ **Improvement yang Diharapkan:**
- Teks menjadi lebih jelas dan tajam
- Noise dan artifacts berkurang
- Kontras dokumen meningkat
- Readability lebih baik

### 📸 **Contoh Hasil:**
- `simple_enhanced_a.png` - Hasil enhancement a.png
- `simple_comparison_a.png` - Perbandingan before/after
- `large_enhanced_document.png` - Hasil dokumen besar

## 🆘 Support dan Help

### 📞 **Jika Mengalami Masalah:**
1. Check log error di terminal
2. Pastikan semua dependencies terinstall
3. Verifikasi format input file
4. Coba dengan dokumen sample terlebih dahulu

### 📝 **Log Files:**
```bash
# Jalankan dengan verbose output
python simple_enhancement_test.py 2>&1 | tee enhancement.log
```

---

**💡 Pro Tip:** Mulai dengan `simple_enhancement_test.py` untuk testing, lalu gunakan CLI tools untuk production!
