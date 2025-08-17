# 🚀 QUICK START: Memperbaiki Dokumen Rusak

## ⚡ Perintah Cepat (Copy & Paste)

### 📄 **Metode Simple - Auto Processing**
```bash
# Jalankan sekali, semua dokumen di folder akan diproses
python simple_enhancement_test.py
```
**Output**: `simple_enhanced_*.png` dan `simple_comparison_*.png`

---

### 🎯 **Metode CLI - Manual Control**
```bash
# Template dasar
python full_document_enhancement.py \
    --input NAMA_FILE_INPUT \
    --output NAMA_FILE_OUTPUT \
    --model ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5

# Contoh konkret
python full_document_enhancement.py \
    --input dokumen_rusak.jpg \
    --output dokumen_diperbaiki.png \
    --model ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5
```

---

### 📊 **Batch Processing - Multiple Files**
```bash
# Untuk semua file JPG
for file in *.jpg; do
    python full_document_enhancement.py \
        --input "$file" \
        --output "enhanced_$file" \
        --model ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5
done

# Untuk semua file PNG
for file in *.png; do
    python full_document_enhancement.py \
        --input "$file" \
        --output "enhanced_$file" \
        --model ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5
done
```

---

## 📁 File Input/Output

### ✅ **Format yang Didukung:**
- **Input**: `.jpg`, `.jpeg`, `.png`, `.tiff`
- **Output**: `.png` (recommended), `.jpg`

### 📤 **File yang Dihasilkan:**
- `enhanced_[nama_file]` - Dokumen hasil perbaikan
- `comparison_[nama_file]` - Perbandingan before/after

---

## 🔧 Parameter Tambahan

```bash
# Dengan parameter lengkap
python full_document_enhancement.py \
    --input dokumen.jpg \
    --output hasil.png \
    --model ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5 \
    --method sliding_window \
    --save-intermediates
```

### 🎛️ **Parameter Options:**
- `--method sliding_window` - Untuk dokumen besar/kompleks
- `--method line_detection` - Untuk dokumen dengan text lines jelas
- `--save-intermediates` - Simpan file proses untuk debugging

---

## 📋 Checklist Before Use

### ✅ **Verifikasi Setup:**
```bash
# 1. Cek model tersedia
ls -la ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5

# 2. Cek file input ada
ls -la dokumen_rusak.jpg

# 3. Test dengan dokumen sample
python simple_enhancement_test.py
```

---

## 🚨 Troubleshooting Quick Fix

### ❌ **Model not found**
```bash
# Cari model alternatif
find . -name "generator.weights.h5" -type f
```

### ❌ **CUDA out of memory**
- Restart Python session
- Proses file satu per satu
- Gunakan dokumen yang lebih kecil

### ❌ **No enhancement result**
```bash
# Coba method alternatif
python full_document_enhancement.py \
    --input dokumen.jpg \
    --output hasil.png \
    --model ./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5 \
    --method sliding_window
```

---

## 🎯 Expected Results

### ✅ **Improvement yang Terlihat:**
- Teks lebih tajam dan jelas
- Noise/artifacts berkurang
- Kontras meningkat
- Readability lebih baik

### 📊 **File Output Example:**
```
simple_enhanced_a.png          # Hasil enhancement
simple_comparison_a.png        # Perbandingan visual
large_enhanced_document.png    # Dokumen besar yang diperbaiki
```

---

## 🔗 Complete Documentation

**📖 Panduan Lengkap**: `MANUAL_PENGGUNAAN.md`
**📋 Table of Contents**: `tableofcontent.md`
**✅ Success Summary**: `DOCUMENT_ENHANCEMENT_SUCCESS_SUMMARY.md`

---

**💡 Pro Tip**: Mulai dengan `simple_enhancement_test.py` untuk testing, kemudian gunakan CLI tools untuk production!
