# Daftar Isi Proyek GAN-HTR

File ini mendokumentasikan struktur direktori dan file utama dalam proyek GAN-HTR. Tujuannya adalah untuk memberikan gambaran umum yang cepat bagi siapa saja yang baru mengenal proyek ini.

## File Utama

- **`.gitignore`**: Menentukan file dan direktori mana yang harus diabaikan oleh sistem kontrol versi Git.
- **`augraphy.md`**: Dokumentasi atau catatan mengenai penggunaan library `augraphy` untuk augmentasi data gambar.
- **`create_iam_file_lists.py`**: Script untuk membuat daftar file (file lists) dari dataset IAM, kemungkinan untuk membaginya menjadi set training, validasi, dan tes.
- **`create_synthetic_degradation_adaptive.py`**: Script untuk menghasilkan degradasi (kerusakan) sintetis pada gambar dokumen dengan metode adaptif.
- **`create_synthetic_degradation.py`**: Versi dasar dari script untuk menghasilkan degradasi sintetis pada gambar.
- **`createBG.md`**: Catatan atau dokumentasi mengenai proses pembuatan gambar latar belakang (background).
- **`dibco_TL_2010.py`**: Script yang berkaitan dengan dataset DIBCO 2010, kemungkinan untuk evaluasi atau pengujian model.
- **`distort_image_augraphy.py`**: Script untuk menerapkan distorsi pada gambar menggunakan library `augraphy`.
- **`distort_image_iam.py`**: Script untuk menerapkan distorsi pada gambar dari dataset IAM.
- **`distort_image_khatt.py`**: Script untuk menerapkan distorsi pada gambar dari dataset Khatt.
- **`download_iam.py`**: Script utilitas untuk mengunduh dataset tulisan tangan IAM.
- **`eval_Dibco_2010.py`**: Script untuk mengevaluasi performa model pada dataset DIBCO 2010.
- **`extract_degradation_patches.py`**: Script untuk mengekstrak potongan-potongan gambar (patches) yang berisi contoh degradasi dari dokumen.
- **`GAN_AHTR.py`**: File utama yang berisi implementasi model Generative Adversarial Network (GAN) untuk Handwritten Text Recognition (HTR).
- **`jnm_GAN_AHTR.py`**: Kemungkinan versi alternatif atau eksperimental dari implementasi GAN-AHTR utama.
- **`LICENSE`**: File lisensi perangkat lunak untuk proyek ini.
- **`logDist_iam.txt`**: File log yang kemungkinan berisi catatan parameter distorsi yang diterapkan pada dataset IAM.
- **`poetry.lock` & `pyproject.toml`**: File-file manajemen dependensi Python menggunakan Poetry.
- **`README.md` & `readme_jnm.md`**: File dokumentasi utama proyek. `readme_jnm.md` mungkin versi personal atau catatan tambahan.
- **`requirements_*.txt`**: File-file yang berisi daftar dependensi Python untuk proyek ini dalam berbagai versi (clean, compatible, updated).
- **`souibgui - ... .pdf`**: Dokumen PDF, kemungkinan besar adalah paper penelitian yang menjadi referensi utama proyek.
- **`tableofcontent.md`**: File ini; berisi daftar isi dan deskripsi file/direktori dalam proyek.
- **`train_khatt_basic_distorted.py`**: Script untuk melatih model menggunakan gambar terdistorsi dari dataset Khatt.
- **`verify_gpu_setup.py`**: Script utilitas untuk memverifikasi apakah setup GPU (CUDA) sudah benar dan siap digunakan.

## Direktori

- **`.git/`**: Direktori internal Git yang berisi semua metadata dan riwayat proyek.
- **`augraphy_cache/`**: Direktori cache untuk library `augraphy`, menyimpan hasil augmentasi agar tidak perlu diproses ulang.
- **`datasets/`**: Direktori utama untuk menyimpan semua dataset, baik yang mentah (`_raw`) maupun yang sudah diproses/di-augmentasi (`_distorted`, `_degraded`).
- **`nan_raw/` & `nan_raw_color/`**: Direktori yang berisi dataset mentah 'nan', terstruktur dalam folder train, test, dan validation.
- **`network/`**: Berisi kode sumber untuk arsitektur model neural network, seperti definisi layer (`layers.py`) dan model itu sendiri (`model.py`).
- **`Sets/`**: Berisi file-file konfigurasi untuk dataset, seperti daftar karakter yang digunakan (`CHAR_LIST`) dan daftar file untuk setiap set data (train, test, validasi).