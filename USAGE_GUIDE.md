# 📖 GAN-HTR Document Enhancement - Panduan Penggunaan

Setelah training model berhasil, berikut adalah **script-script yang bisa Anda gunakan** untuk restorasi/enhancement dokumen:

## 🎯 **Script Utama yang Tersedia**

### 1. **`document_enhancer.py`** - Enhancement Single Image
**Script utama untuk enhancement satu gambar**

```bash
# Cara penggunaan dasar
poetry run python document_enhancer.py --input path/to/degraded_image.jpg

# Dengan custom output path
poetry run python document_enhancer.py --input image.jpg --output enhanced_image.png

# Dengan custom model
poetry run python document_enhancer.py --input image.jpg --model path/to/model.h5

# Tanpa comparison plot
poetry run python document_enhancer.py --input image.jpg --no-comparison
```

**Contoh konkret:**
```bash
# Enhancement file test yang sudah dicoba
poetry run python document_enhancer.py --input datasets/nan_distorted/test/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg

# Enhancement dengan output custom
poetry run python document_enhancer.py --input my_degraded_document.jpg --output my_enhanced_result.png
```

### 2. **`batch_enhancer.py`** - Enhancement Multiple Images
**Script untuk memproses banyak gambar sekaligus**

```bash
# Enhancement seluruh folder
poetry run python batch_enhancer.py --input_dir datasets/nan_distorted/test/ --output_dir enhanced_results/

# Dengan custom model
poetry run python batch_enhancer.py --input_dir input_images/ --output_dir output_images/ --model custom_model.h5

# Hanya format tertentu
poetry run python batch_enhancer.py --input_dir images/ --output_dir enhanced/ --formats "*.jpg" "*.png"
```

### 3. **`simple_test_model.py`** - Testing & Analysis
**Script untuk testing detail dengan metrics**

```bash
# Test dengan file spesifik (yang sudah berhasil)
poetry run python simple_test_model.py
```

---

## 🚀 **Quick Start Guide**

### **Step 1: Enhancement Single Image**
```bash
cd /home/lambda_one/tesis/GAN-HTR

# Test dengan file yang sudah berhasil
poetry run python document_enhancer.py --input datasets/nan_distorted/test/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg
```

### **Step 2: Enhancement Your Own Images**
```bash
# Ganti dengan path gambar Anda sendiri
poetry run python document_enhancer.py --input /path/to/your/degraded_document.jpg --output /path/to/enhanced_result.png
```

### **Step 3: Batch Processing** 
```bash
# Untuk memproses banyak gambar sekaligus
poetry run python batch_enhancer.py --input_dir /path/to/degraded_images/ --output_dir /path/to/enhanced_results/
```

---

## 📊 **Output yang Dihasilkan**

### **Single Enhancement (`document_enhancer.py`):**
- `enhanced_[filename].png` - Hasil enhancement
- `enhanced_[filename]_comparison.png` - Perbandingan visual
- Console output dengan info ukuran dan status

### **Batch Enhancement (`batch_enhancer.py`):**
- Folder output dengan struktur yang sama seperti input
- Progress bar untuk tracking
- Summary report jumlah sukses/error

### **Testing (`simple_test_model.py`):**
- `test_results/enhancement_test.png` - Visual comparison
- `test_results/enhanced_output.png` - Hasil enhancement
- Metrics: PSNR, SSIM, improvement statistics

---

## ⚙️ **Model Information**

**Model yang digunakan:**
- **Path:** `checkpoints/improved_model_20250814_051937/model_epoch_15_generator.weights.h5`
- **Architecture:** U-Net Generator
- **Input Size:** 128x128 grayscale
- **Training Data:** 3,839 aligned pairs (NaN dataset)
- **Performance:** SSIM improvement +89% (0.32 → 0.60)

**Supported Image Formats:**
- `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`
- Grayscale conversion otomatis
- Auto-resize ke ukuran asli setelah enhancement

---

## 🔧 **Troubleshooting**

### **Error "Model not found":**
```bash
# Pastikan model path benar
ls -la checkpoints/improved_model_20250814_051937/

# Atau gunakan custom model path
poetry run python document_enhancer.py --input image.jpg --model /full/path/to/model.h5
```

### **Error "Image not found":**
```bash
# Pastikan path gambar benar
ls -la /path/to/your/image.jpg

# Gunakan absolute path
poetry run python document_enhancer.py --input /home/lambda_one/tesis/GAN-HTR/datasets/nan_distorted/test/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg
```

### **Memory Issues:**
- Model memerlukan GPU dengan minimal 4GB VRAM
- Untuk batch processing, proses satu per satu (tidak parallel)
- Gambar besar akan di-resize otomatis

---

## 📋 **Recommended Workflow**

1. **Single Test:** Mulai dengan `document_enhancer.py` untuk test satu gambar
2. **Verify Quality:** Check hasil enhancement dan comparison plot
3. **Batch Process:** Jika hasil bagus, gunakan `batch_enhancer.py` untuk banyak gambar
4. **Analysis:** Gunakan `simple_test_model.py` untuk analisis detail metrics

---

## 🎯 **Script Recommendation**

**Untuk penggunaan sehari-hari:** ➜ **`document_enhancer.py`**
- Mudah digunakan
- Output clear dengan visualization
- Support custom paths
- Auto-generate output filename

**Untuk processing banyak file:** ➜ **`batch_enhancer.py`**
- Efficient batch processing  
- Progress tracking
- Error handling
- Maintain folder structure

**Untuk research/analysis:** ➜ **`simple_test_model.py`**
- Detailed metrics (PSNR, SSIM)
- Visual analysis tools
- Comparison dengan ground truth

---

**Ready to use! 🚀 Model sudah trained dan script sudah teruji.**
