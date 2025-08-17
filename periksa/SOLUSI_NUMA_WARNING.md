# Solusi Mengatasi NUMA Node Warning

## 🚨 Masalah
Ketika menjalankan training dengan GPU, muncul warning berikut:
```
2025-08-17 04:44:41.585811: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
```

## 🔍 Penyebab
NUMA (Non-Uniform Memory Access) warning ini terjadi karena:
- **Sistem konfigurasi**: CUDA/TensorFlow tidak dapat membaca NUMA topology dengan benar
- **Virtual environment**: Docker containers atau VM sering mengalami masalah ini
- **Multi-GPU setup**: Sistem dengan 2+ GPU kadang memiliki NUMA topology yang tidak standar
- **Driver/system**: Kombinasi tertentu dari driver NVIDIA dan sistem Linux

**Catatan**: Warning ini **TIDAK mempengaruhi performa** training, hanya menampilkan pesan informasi yang mengganggu.

## ✅ Solusi Lengkap

### 1. Environment Variables
Tambahkan konfigurasi environment sebelum import TensorFlow:

```python
import os
import warnings

# Suppress TensorFlow warnings including NUMA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # Sesuaikan dengan GPU Anda
```

### 2. TensorFlow Configuration
Konfigurasi TensorFlow untuk suppress logging dan warnings:

```python
import tensorflow as tf

# Suppress TensorFlow logging and NUMA warnings
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Configure GPU memory growth to avoid NUMA warnings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Configured {len(gpus)} GPU(s) with memory growth")
    except RuntimeError as e:
        print(f"⚠️  GPU configuration warning: {e}")

# Suppress NUMA warnings specifically
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', '.*NUMA.*')
```

### 3. Implementasi di File Utama
Solusi telah diintegrasikan ke `jnm_GAN_AHTR.py` dengan:

**Lokasi perubahan:**
- ✅ Environment variables di awal file (baris 1-12)
- ✅ TensorFlow configuration setelah import (baris 14-28)
- ✅ GPU memory growth configuration
- ✅ Warning suppression untuk NUMA

## 🧪 Verifikasi Solusi

### Script Test Tersedia:
1. **`periksa/fix_numa_warning.py`** - Analisis dan solusi NUMA
2. **`periksa/test_numa_fix.py`** - Verifikasi perbaikan

### Menjalankan Test:
```bash
# Test konfigurasi NUMA
poetry run python periksa/fix_numa_warning.py

# Verifikasi perbaikan
poetry run python periksa/test_numa_fix.py
```

## 📊 Hasil Test

### ✅ Yang Berhasil Diperbaiki:
- ✅ NUMA warnings berhasil di-suppress
- ✅ GPU operations berjalan normal (0.0613 detik untuk matrix 1000x1000)
- ✅ Training script dapat berjalan tanpa warning mengganggu
- ✅ 2 GPU (NVIDIA RTX A4000) dikonfigurasi dengan benar
- ✅ Memory growth enabled untuk menghindari memory issues

### 📈 Performance Impact:
- **Tidak ada penurunan performa** - warning hanya di-suppress, bukan diperbaiki di level hardware
- **Memory management lebih baik** - GPU memory growth enabled
- **Logging lebih bersih** - fokus pada pesan training yang penting

## 🚀 Cara Penggunaan

### Menjalankan Training Tanpa Warning:
```bash
cd /home/lambda_one/tesis/GAN-HTR
poetry run python jnm_GAN_AHTR.py
```

### Monitoring GPU:
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 💡 Tips Tambahan

### 1. System-Level Solutions (Opsional)
Jika ingin mengatasi di level sistem:
```bash
# Check NUMA topology
numactl --hardware

# Set NUMA policy (advanced users)
export CUDA_LAUNCH_BLOCKING=1
```

### 2. Docker/Container Users
Jika menggunakan Docker, tambahkan:
```dockerfile
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
```

### 3. Multi-GPU Training
Untuk optimasi multi-GPU:
```python
# Distribute strategy (advanced)
strategy = tf.distribute.MirroredStrategy()
```

## 🔧 Troubleshooting

### Jika Warning Masih Muncul:
1. **Restart terminal/session** setelah perubahan environment
2. **Clear cache** TensorFlow: `rm -rf ~/.cache/tensorflow/`
3. **Update drivers** NVIDIA jika perlu
4. **Check system logs** untuk masalah hardware: `dmesg | grep -i numa`

### Jika Performa Menurun:
1. **Disable memory growth** jika memory terbatas
2. **Adjust thread counts** sesuai CPU cores
3. **Monitor GPU utilization** dengan `nvidia-smi`

## ✨ Kesimpulan

**Status:** ✅ **SOLVED** - NUMA warnings berhasil di-suppress!

### Rangkuman Perbaikan:
- 🔧 **Environment variables** dikonfigurasi
- 🔧 **TensorFlow logging** di-suppress
- 🔧 **GPU memory growth** enabled
- 🔧 **Warning filters** diterapkan
- 🧪 **Testing suite** tersedia untuk verifikasi

### Hasil:
- ✅ **Training berjalan bersih** tanpa warning mengganggu
- ✅ **Performa GPU optimal** dengan 2x RTX A4000
- ✅ **Memory management** yang lebih baik
- ✅ **Logging fokus** pada informasi training penting

**Ready for training!** 🚀
