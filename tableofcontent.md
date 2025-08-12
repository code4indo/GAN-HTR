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

### jnm_GAN_AHTR.py: Dokumentasi Lengkap

Skrip ini merupakan implementasi inti untuk melatih sebuah *Generative Adversarial Network* (GAN) yang dirancang khusus untuk **memperbaiki (enhancement) kualitas gambar tulisan tangan yang terdegradasi atau rusak**. Tujuannya adalah untuk mengubah gambar input yang berkualitas buruk menjadi gambar yang bersih, jelas, dan dioptimalkan untuk dikenali oleh sistem *Handwriting Text Recognition* (HTR).

#### Cara Kerja dan Arsitektur

Skrip ini menggunakan pendekatan GAN yang canggih dengan dua diskriminator untuk memastikan kualitas hasil dari dua aspek: visual dan keterbacaan.

1.  **Generator**:
    *   **Arsitektur**: Menggunakan model **U-Net**.
    *   **Tugas**: Menerima gambar tulisan tangan yang rusak dan mencoba merekonstruksi atau "membersihkannya" untuk menghasilkan citra yang ideal.

2.  **Discriminators (Diskriminator)**:
    *   **Diskriminator 1 (Penilai Visual)**:
        *   **Arsitektur**: Jaringan Konvolusi (CNN) sederhana.
        *   **Tugas**: Menilai apakah gambar yang dihasilkan oleh Generator terlihat "nyata" dan berkualitas tinggi secara visual, seolah-olah itu adalah pindaian dokumen asli yang bersih.
    *   **Diskriminator 2 (Penilai Keterbacaan)**:
        *   **Arsitektur**: Model **CRNN (Convolutional Recurrent Neural Network)**, yang merupakan arsitektur standar untuk HTR.
        *   **Tugas**: Mencoba membaca dan mengenali teks dari gambar yang dihasilkan oleh Generator. Ini memaksa Generator untuk tidak hanya membuat gambar yang indah secara visual, tetapi juga yang teksnya benar-benar dapat dibaca dan dikenali dengan akurat.

#### Proses Pelatihan

Generator dilatih untuk "menipu" kedua diskriminator secara bersamaan. Kerugian (loss) dihitung dari tiga komponen:
1.  Seberapa baik Generator menipu diskriminator visual.
2.  Seberapa mirip gambar hasil Generator dengan gambar asli yang bersih (loss `binary_crossentropy`).
3.  Seberapa baik Generator menipu diskriminator CRNN (loss `ctc_loss_lambda_func`).

#### Output yang Dihasilkan

Saat skrip dijalankan, ia akan menghasilkan file dan direktori berikut, terutama di dalam direktori `ResultGanS_iam_OP/` (nama direktori ditentukan oleh variabel `scenario`).

##### Struktur Direktori Output

```
/home/lambda_one/tesis/GAN-HTR/
├── ResultGanS_iam_OP/
│   ├── epoch0/
│   │   ├── weights/
│   │   │   ├── discriminator_weights.h5
│   │   │   ├── gan_weights.h5
│   │   │   ├── generator_weights.h5
│   │   │   └── rcnn_weights.h5
│   │   ├── a01-000u-00.png
│   │   └── ... (gambar evaluasi lainnya)
│   │
│   ├── epoch4/
│   │   └── ... (struktur yang sama diulang)
│   └── ...
│
├── charlist.txt
└── ... (file proyek lainnya)
```

##### Rincian File Output

1.  **Bobot Model (di dalam `epochX/weights/`)**:
    *   `generator_weights.h5`: **Output paling penting**. Ini adalah model Generator terlatih yang dapat digunakan secara mandiri untuk membersihkan gambar tulisan tangan baru.
    *   `discriminator_weights.h5`: Bobot untuk diskriminator visual.
    *   `rcnn_weights.h5`: Bobot untuk diskriminator pengenal teks (CRNN).
    *   `gan_weights.h5`: Menyimpan keadaan dari seluruh model GAN gabungan.

2.  **Gambar Hasil Evaluasi (di dalam `epochX/`)**:
    *   Untuk setiap gambar dalam set validasi, sebuah file PNG akan disimpan (misal: `a01-000u-00.png`).
    *   Gambar ini adalah gabungan vertikal dari 3 citra untuk perbandingan langsung:
        1.  **Atas**: Gambar asli yang rusak (input).
        2.  **Tengah**: Gambar yang telah diperbaiki oleh Generator.
        3.  **Bawah**: Gambar *ground truth* (versi asli yang bersih).

3.  **File `charlist.txt`**:
    *   Dibuat di direktori root proyek.
    *   Berisi daftar semua karakter unik yang digunakan oleh model CRNN untuk mengenali teks.

##### Output di Konsol

Selama eksekusi, terminal akan menampilkan:
-   Pesan status inisialisasi model (`generator creation...`, `discriminator 1 creation...`).
-   Informasi kemajuan per-epoch (`Epoch 1`, `Epoch 2`, ...).
-   Progress bar dari `tqdm` yang menunjukkan pemrosesan data dalam satu epoch.
-   Log pelatihan dari Keras, termasuk metrik `loss` dan `accuracy`.

#### Kebutuhan Dataset untuk Pelatihan

Agar skrip dapat berjalan dengan lancar, dataset IAM harus disiapkan dengan struktur dan format spesifik sebagai berikut:

1.  **Dataset Gambar Berpasangan**: Skrip membutuhkan pasangan gambar yang terdiri dari versi bersih dan versi rusak, dengan nama file yang identik.
    *   **Gambar Asli/Bersih (Ground Truth)**: Ditempatkan di `datasets/iam_raw/`. Contoh: `datasets/iam_raw/a01-000u-00.png`.
    *   **Gambar Rusak/Terdegradasi (Input)**: Ditempatkan di `datasets/iam_distorted/`. Contoh: `datasets/iam_distorted/a01-000u-00.png`.

2.  **File Konfigurasi dan Teks (dalam direktori `Sets/`)**:
    *   **`Sets/lines.txt`**: Berisi transkripsi teks untuk setiap gambar, digunakan untuk melatih model pengenalan teks (CRNN).
    *   **`Sets/list_train_iam.txt`**: Daftar nama file gambar yang digunakan untuk data pelatihan.
    *   **`Sets/list_valid_iam.txt`**: Daftar nama file gambar yang digunakan untuk data validasi.
    *   **`Sets/CHAR_LIST`**: Daftar semua karakter unik yang akan dikenali oleh model.

Struktur direktori yang diperlukan adalah:
```
/home/lambda_one/tesis/GAN-HTR/
├── datasets/
│   ├── iam_raw/
│   │   └── ... (gambar-gambar asli)
│   └── iam_distorted/
│       └── ... (gambar-gambar rusak)
└── Sets/
    ├── CHAR_LIST
    ├── lines.txt
    ├── list_train_iam.txt
    └── list_valid_iam.txt
```

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