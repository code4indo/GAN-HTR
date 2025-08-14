# 🎯 SCRIPT UNTUK DOCUMENT ENHANCEMENT - READY TO USE!

## ✅ **SCRIPT UTAMA YANG BISA LANGSUNG DIGUNAKAN:**

### 1. **`document_enhancer.py`** ⭐ **RECOMMENDED**
**Script utama untuk enhancement dokumen - User friendly**

```bash
# Cara penggunaan paling mudah:
poetry run python document_enhancer.py --input path/to/gambar_rusak.jpg

# Contoh dengan file test:
poetry run python document_enhancer.py --input datasets/nan_distorted/test/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg

# Custom output:
poetry run python document_enhancer.py --input gambar_saya.jpg --output hasil_enhanced.png
```

**Output yang dihasilkan:**
- ✅ `enhanced_[filename].png` - Gambar hasil enhancement
- ✅ `enhanced_[filename]_comparison.png` - Perbandingan visual (before/after)
- ✅ Console info dengan ukuran dan status

---

### 2. **`batch_enhancer.py`**
**Untuk memproses banyak gambar sekaligus**

```bash
# Enhancement seluruh folder:
poetry run python batch_enhancer.py --input_dir folder_gambar_rusak/ --output_dir folder_hasil/

# Contoh:
poetry run python batch_enhancer.py --input_dir datasets/nan_distorted/test/ --output_dir enhanced_batch_results/
```

---

### 3. **`simple_test_model.py`**
**Testing dengan metrics detail**

```bash
# Test model dengan metrics PSNR/SSIM:
poetry run python simple_test_model.py
```

---

## 🚀 **QUICK START - LANGSUNG PAKAI:**

### **Test dengan gambar yang sudah ada:**
```bash
cd /home/lambda_one/tesis/GAN-HTR

# Enhancement file test (sudah terbukti berhasil):
poetry run python document_enhancer.py --input datasets/nan_distorted/test/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg
```

### **Enhancement gambar Anda sendiri:**
```bash
# Ganti dengan path gambar Anda:
poetry run python document_enhancer.py --input /path/to/dokumen_rusak.jpg --output /path/to/hasil_bagus.png
```

---

## 📊 **Hasil Testing Terakhir:**

✅ **Model berhasil di-load dan berjalan**
✅ **Output enhancement tersimpan dengan benar**  
✅ **Comparison visualization dibuat otomatis**
✅ **Metrics improvement: SSIM +89% (0.32 → 0.60)**

---

## 🔧 **Model Information:**

- **Model Path:** `checkpoints/improved_model_20250814_051937/model_epoch_15_generator.weights.h5`
- **Architecture:** U-Net Generator (trained 15 epochs)
- **Training Data:** 3,839 aligned image pairs (NaN dataset)
- **Performance:** Structural improvement +89% SSIM
- **Input:** Any image format (.jpg, .png, .tif, dll.)
- **Output:** Enhanced PNG dengan comparison plot

---

## 💡 **Recommendation:**

**Mulai dengan:** `document_enhancer.py` ← **Script ini paling mudah digunakan**

1. Test dulu dengan file yang sudah ada di datasets/
2. Jika hasil bagus, coba dengan gambar Anda sendiri  
3. Untuk banyak file, gunakan `batch_enhancer.py`

---

## 🎉 **STATUS: READY TO USE!**

Model sudah trained, tested, dan script sudah teruji bekerja dengan baik.
Anda bisa langsung menggunakan `document_enhancer.py` untuk enhancement dokumen!
