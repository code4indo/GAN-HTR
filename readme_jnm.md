# Panduan Memulai Proyek GAN-HTR dengan Poetry

![GAN-HTR](https://img.shields.io/static/v1?label=Poetry&message=GAN-HTR&color=blue&size=30)

Selamat datang di proyek **GAN-HTR** (Generative Adversarial Network for Handwritten Text Recognition)!

File ini berisi panduan lengkap untuk setup proyek menggunakan **Poetry**, sebuah tool modern untuk manajemen dependensi dan virtual environment di Python.

## 1. 🚀 Prasyarat

Sebelum memulai, pastikan Anda telah menginstall **Poetry** di sistem Anda. Jika belum, ikuti instruksi di [situs resmi Poetry](https://python-poetry.org/docs/#installation).

**Verifikasi instalasi Poetry:**
```bash
poetry --version
```

## 2. ⚙️ Instalasi Proyek

Dengan Poetry, seluruh proses pembuatan virtual environment dan instalasi dependensi dapat dilakukan dengan satu perintah.

### 2.1. Clone Repository
Jika Anda belum melakukannya, clone repository ini ke mesin lokal Anda.
```bash
git clone <URL_REPOSITORY_ANDA>
cd GAN-HTR
```

### 2.2. Install Dependensi
Poetry akan membaca file `pyproject.toml`, membuat virtual environment khusus untuk proyek ini, dan menginstall semua dependensi yang diperlukan.

```bash
poetry install
```
Perintah ini akan:
- ✅ Membuat virtual environment secara otomatis.
- ✅ Menginstall semua library yang tercantum di `pyproject.toml`.
- ✅ Membuat file `poetry.lock` untuk memastikan instalasi yang konsisten di setiap mesin.

## 3. 激活 Virtual Environment

Untuk bekerja di dalam lingkungan proyek, aktifkan *shell* yang disediakan oleh Poetry.

```bash
poetry shell
```
Perintah ini akan 'memasukkan' Anda ke dalam lingkungan virtual proyek. Setelah dijalankan, prompt terminal Anda akan berubah, menandakan bahwa Anda sekarang berada di dalam virtual environment proyek. Anda dapat menjalankan perintah Python dan skrip secara langsung.

**Alternatif: Menggunakan Perintah `source`**
Jika Anda lebih suka mengaktifkan virtual environment secara manual (misalnya, untuk integrasi dengan IDE atau skrip), Anda bisa menggunakan perintah `source` dengan path ke skrip `activate` di virtual environment Poetry.

1.  **Temukan path virtual environment Anda:**
    ```bash
    poetry env info --path
    ```
    Output akan menunjukkan path seperti `/home/lambda_one/.cache/pypoetry/virtualenvs/gan-htr-DgUpKV58-py3.10` (path ini bisa berbeda di sistem Anda).

2.  **Aktifkan virtual environment:**
    ```bash
    source $(poetry env info --path)/bin/activate
    ```
    Atau, jika Anda sudah tahu path-nya:
    ```bash
    source /home/lambda_one/.cache/pypoetry/virtualenvs/gan-htr-DgUpKV58-py3.10/bin/activate
    ```
    Setelah diaktifkan, prompt terminal Anda juga akan berubah.

**Untuk keluar dari shell, cukup ketik `exit`.**

## 4. 🔬 Verifikasi Instalasi

Setelah dependensi terinstall dan *shell* aktif, verifikasi bahwa semua library utama berfungsi dengan baik.

### 4.1. Verifikasi Cepat
Jalankan perintah berikut untuk memastikan library utama dapat di-import.
```bash
python -c "import tensorflow as tf, cv2, numpy as np, scipy, h5py; print('✅ Semua library utama berhasil diimport!')"
```

### 4.2. Verifikasi Versi Library
```bash
python -c "
import tensorflow as tf
import numpy as np
import cv2
import scipy
print('✅ TensorFlow:', tf.__version__)
print('✅ NumPy:', np.__version__)
print('✅ OpenCV:', cv2.__version__)
print('✅ SciPy:', scipy.__version__)
"
```

### 4.3. Menjalankan Skrip Tanpa Aktivasi Shell
Jika Anda tidak ingin mengaktifkan `poetry shell`, Anda bisa menjalankan skrip menggunakan `poetry run`.

```bash
poetry run python -c "import tensorflow as tf; print('✅ TensorFlow version:', tf.__version__)"
```

## 5. 🎮 Setup Khusus untuk GPU

Proyek ini dikonfigurasi untuk menggunakan TensorFlow dengan dukungan GPU. Poetry akan mengurus instalasi versi yang tepat.

### 5.1. Verifikasi Driver dan CUDA
Pastikan driver NVIDIA dan CUDA toolkit sudah terinstall di sistem Anda.
```bash
# Cek driver NVIDIA
nvidia-smi

# Cek versi CUDA
nvcc --version
```

### 5.2. Verifikasi TensorFlow GPU
Jalankan skrip berikut untuk memastikan TensorFlow dapat mendeteksi dan menggunakan GPU Anda.

```bash
poetry run python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('CUDA built:', tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices('GPU')
print(f'GPU devices found: {len(gpus)}')

if gpus:
    for i, gpu in enumerate(gpus):
        print(f'  GPU {i}: {gpu}')
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f'    Device name: {details.get(\"device_name\", \"Unknown\")}')
            print(f'    Compute capability: {details.get(\"compute_capability\", \"Unknown\")}')
        except:
            print('    Details not available')
    
    # Test komputasi sederhana di GPU
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([1.0, 2.0, 3.0])
            b = tf.constant([4.0, 5.0, 6.0])
            c = tf.add(a, b)
            print(f'✅ GPU computation test successful: {c.numpy()}')
    except Exception as e:
        print(f'❌ GPU computation failed: {e}')
else:
    print('⚠️  No GPU detected - TensorFlow will use CPU.')
"
```

### 5.3. Troubleshooting GPU: `libdevice.10.bc not found`
Jika Anda mendapatkan error `libdevice.10.bc not found`, ini berarti TensorFlow tidak dapat menemukan CUDA library. Solusi paling umum adalah membuat symbolic link.

1.  **Temukan file `libdevice.10.bc`:**
    ```bash
    sudo find / -name libdevice.10.bc
    ```
    Catat path yang ditemukan (misalnya: `/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc`).

2.  **Buat symbolic link di direktori proyek:**
    ```bash
    # Pastikan Anda berada di direktori root proyek GAN-HTR
    ln -s /path/ke/libdevice.10.bc .
    ```
    Ganti `/path/ke/libdevice.10.bc` dengan path yang Anda temukan.

## 6. 📂 Struktur Proyek (Versi Poetry)

```
GAN-HTR/
├── 📜 GAN_AHTR.py                       # Script utama untuk training GAN-HTR
├── 🎯 train_khatt_basic_distorted.py    # Training text recognition
├── 📊 eval_Dibco_2010.py                # Evaluasi untuk document binarization
├── 🎨 distort_image_khatt.py            # Generate distorted images
├── 🔧 verify_gpu_setup.py               # Skrip verifikasi GPU
├── 📂 network/                          # Arsitektur neural network
│   ├── layers.py
│   └── model.py
├── 📂 Sets/                             # Konfigurasi dataset
│   ├── CHAR_LIST
│   └── ...
├── 📄 pyproject.toml                    # File konfigurasi proyek dan dependensi (untuk Poetry) ⭐
├── 📄 poetry.lock                       # File lock untuk versi dependensi yang presisi
├── 📖 README.md                         # Dokumentasi utama
└── 📖 readme_jnm.md                     # Panduan setup (file ini)
```

## 7. 🚀 Menjalankan Skrip Proyek

Gunakan `poetry run` untuk menjalankan skrip-skrip utama proyek.

```bash
# Training GAN-HTR
poetry run python GAN_AHTR.py

# Evaluasi DIBCO
poetry run python eval_Dibco_2010.py

# Membuat gambar distorsi
poetry run python distort_image_khatt.py
```
Untuk memonitor penggunaan GPU, buka terminal lain dan jalankan:
```bash
watch -n 1 nvidia-smi
```

## 8. 🔧 Troubleshooting dengan Poetry

| Masalah | Solusi |
|---------|--------|
| ❌ Perintah `python` tidak ditemukan | Pastikan Anda sudah mengaktifkan shell dengan `poetry shell`, atau gunakan `poetry run`. |
| 🔄 Ingin mengupdate library | Gunakan `poetry update <nama_package>` atau `poetry update` untuk semua. |
| 🔍 Ingin melihat info environment | `poetry env info` akan menampilkan path ke virtual environment dan versi Python. |
| 💥 Environment rusak | Hapus environment lama dan install ulang: `poetry env remove python && poetry install`. |
| 📦 Menambah library baru | `poetry add <nama_package>` |

---

## 9. 📥 Men-download Dataset dari Kaggle

Untuk mengelola dataset dari Kaggle secara efisien, Anda dapat menggunakan Kaggle API langsung dari dalam proyek ini.

### 9.1. Install Library Kaggle
Tambahkan library `kaggle` ke dalam proyek Anda menggunakan Poetry.
```bash
poetry add kaggle
```

### 9.2. Setup Kaggle API Key
1.  **Buat API Token di Kaggle:**
    - Buka situs Kaggle dan masuk ke akun Anda.
    - Pergi ke halaman profil Anda, lalu klik **"Account"**.
    - Gulir ke bawah hingga menemukan bagian **"API"**.
    - Klik tombol **"Create New API Token"**. Ini akan mengunduh file `kaggle.json`.

2.  **Simpan API Key:**
    - Buat direktori `.kaggle` di folder *home* Anda jika belum ada.
    - Pindahkan file `kaggle.json` yang baru saja diunduh ke direktori tersebut.

    ```bash
    # Buat direktori (jika belum ada)
    mkdir -p ~/.kaggle

    # Pindahkan file (ganti ~/Downloads/kaggle.json dengan path unduhan Anda)
    mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

    # Atur izin file agar aman
    chmod 600 ~/.kaggle/kaggle.json
    ```

### 9.3. Download Dataset
1.  **Cari Perintah API di Halaman Dataset:**
    - Buka halaman dataset yang ingin Anda unduh di Kaggle.
    - Klik ikon tiga titik (menu) dan pilih **"Copy API command"**.

2.  **Jalankan Perintah Download:**
    - Gunakan `poetry run` untuk menjalankan perintah yang telah Anda salin. Contoh:
    ```bash
    # Contoh untuk dataset 'iam-handwriting-database'
    poetry run kaggle datasets download -d landlord/iam-handwriting-database -p datasets/
    ```
    - Opsi `-p datasets/` akan mengunduh file langsung ke folder `datasets` di proyek Anda.

### 9.4. Unzip Dataset
Dataset dari Kaggle biasanya dalam format `.zip`. Anda bisa mengekstraknya menggunakan perintah `unzip`.

```bash
# Masuk ke folder datasets
cd datasets/

# Unzip file (ganti nama file sesuai dengan yang diunduh)
unzip iam-handwriting-database.zip

# (Opsional) Hapus file zip setelah diekstrak
rm iam-handwriting-database.zip
```

---

## 10. 📥 (Cara Baru) Men-download Dataset Hugging Face dari Kaggle

Kaggle sekarang menyediakan cara yang lebih modern untuk men-download dataset, terutama yang terintegrasi dengan Hugging Face, menggunakan library `kagglehub`.

### 10.1. Install Dependensi
Tambahkan library `kagglehub` dengan *extra* `hf-datasets` ke proyek Anda.
```bash
poetry add "kagglehub[hf-datasets]"
```
Perintah ini akan menginstall `kagglehub` dan `datasets` dari Hugging Face.

### 10.2. Buat Skrip Download
Saya telah membuatkan skrip `download_khatt_dataset.py` untuk Anda. Skrip ini akan men-download dataset `nizarcharrada/khattarabic` dan menyimpannya di *cache* lokal (`~/.cache/kagglehub`).

### 10.3. Jalankan Skrip Download
Pastikan API key Kaggle Anda sudah ter-setup (lihat bagian 9.2), lalu jalankan skrip berikut:
```bash
poetry run python download_khatt_dataset.py
```
Skrip ini akan:
1.  Mengunduh dataset jika belum ada di *cache*.
2.  Me-load dataset sebagai objek `DatasetDict` dari Hugging Face.
3.  Menampilkan struktur dataset dan beberapa contoh data.

Dengan cara ini, Anda tidak perlu mengelola file `.zip` secara manual. Library `datasets` akan menangani semua proses di latar belakang.

---

## 🎉 Proyek Siap Digunakan!

> **Happy coding!** 🚀
