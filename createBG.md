Tentu, ini adalah contoh prompt detail yang merangkum seluruh logika proses, yang bisa Anda berikan kepada seorang developer, AI, atau untuk dokumentasi proyek Anda.

---

### ## Prompt Detail: Logika Proses Pembuatan Dataset Degradasi Sintetis

**Tujuan Utama:** Membuat sebuah dataset gambar baris teks terdegradasi yang **realistis dan beragam** secara otomatis. Dataset ini akan digunakan untuk melatih model Machine Learning (misalnya, OCR) agar tangguh terhadap berbagai jenis kerusakan pada dokumen nyata.

**Input Utama:**
1.  `sumber_degradasi.jpg`: Sebuah file gambar tunggal (resolusi tinggi) dari sebuah dokumen yang mengalami degradasi parah dan beragam (misalnya, ada noda, lipatan, sobekan, noise, dan pudar di area yang berbeda).
2.  `dataset_bersih/`: Sebuah direktori berisi ribuan gambar baris teks yang sudah bersih dan ter-binarisasi (teks hitam, latar belakang transparan/putih). Setiap file gambar memiliki nama yang sesuai dengan isinya (misalnya, `ini_adalah_contoh.png`).

---

### ## Logika Proses Langkah-demi-Langkah

**Logika Inti:** Kita tidak akan menggunakan `sumber_degradasi.jpg` sebagai satu latar belakang statis. Sebaliknya, kita akan memperlakukannya sebagai **"palet" atau "tambang" tekstur degradasi**. Kita akan "menambang" banyak potongan kecil dari gambar ini untuk menciptakan latar belakang yang unik bagi setiap baris teks.

**Langkah 1: Persiapan Input**
* Muat gambar `sumber_degradasi.jpg` ke dalam memori.
* Konversi gambar sumber degradasi ke mode Grayscale untuk mensimulasikan kertas tua dan tinta yang lebih realistis.
* Dapatkan daftar semua file gambar baris teks dari direktori `dataset_bersih/`.

**Langkah 2: Proses Cropping Acak (Inti Logika)**
* Untuk setiap gambar baris teks bersih di `dataset_bersih/`:
    * Tentukan dimensi baris teks tersebut (misalnya, `lebar_teks` x `tinggi_teks`).
    * Tentukan dimensi target untuk potongan latar belakang (`crop`). Ukuran `crop` harus **sedikit lebih besar** dari baris teks untuk memberikan ruang bagi pergeseran posisi (`jitter`).
        * `lebar_crop = lebar_teks + padding_horizontal` (misal, padding 20-50 piksel)
        * `tinggi_crop = tinggi_teks + padding_vertikal` (misal, padding 10-30 piksel)
    * Pilih koordinat `(x, y)` secara **acak** dari dalam gambar `sumber_degradasi.jpg`. Ini adalah titik awal untuk `crop`. Pastikan titik ini dipilih sedemikian rupa sehingga `crop` tidak keluar dari batas gambar sumber. 
    * Lakukan `crop` dari `sumber_degradasi.jpg` pada posisi `(x, y)` dengan ukuran `lebar_crop` x `tinggi_crop`. Potongan ini sekarang menjadi **latar belakang degradasi yang unik**.

**Langkah 3: Proses Augmentasi dan Penggabungan**
* Ambil gambar baris teks bersih yang sedang diproses.
* Tentukan posisi penempatan baris teks di dalam `crop` latar belakang secara **acak** di dalam area `padding`. Ini disebut **"jitter"** dan penting untuk variasi.
* Gabungkan (overlay) gambar baris teks ke atas `crop` latar belakang. **PENTING:** Jangan hanya menempelkannya. Gunakan **Mode Pencampuran (Blending Mode)** seperti `Multiply` atau `Overlay`.
    * **Mengapa Blending Mode?** Mode `Multiply` akan membuat area putih pada gambar teks menjadi transparan dan menggelapkan piksel di bawah area hitam, sehingga tekstur kertas dari latar belakang tetap **terlihat menembus tinta**. Ini sangat krusial untuk realisme.
    * 

**Langkah 4: Penyimpanan Hasil**
* Simpan gambar hasil penggabungan ke direktori output (misalnya, `dataset_sintetis/`). Nama filenya harus tetap sama dengan file teks bersih aslinya untuk menjaga keterlacakan (misalnya, `ini_adalah_contoh.png`).
* Simpan **"ground truth"** atau labelnya. Buat file `labels.csv` yang berisi dua kolom: `nama_file` dan `teks_asli`. Ini memastikan setiap gambar sintetis memiliki label yang benar untuk pelatihan model.

**Langkah 5: Iterasi**
* Ulangi Langkah 2 hingga 4 untuk **semua** gambar baris teks di `dataset_bersih/`. Pastikan setiap iterasi menghasilkan `crop` latar belakang dari posisi acak yang baru untuk memaksimalkan keragaman.

---

### ## Output yang Diharapkan

1.  **Direktori `dataset_sintetis/`:** Berisi gambar-gambar baris teks yang kini terlihat terdegradasi secara alami dan bervariasi.
2.  **File `labels.csv`:** Sebuah file pemetaan antara nama file di `dataset_sintetis/` dengan konten teks aslinya.

---
**Catatan Penting:**
* **Hindari Pengulangan:** Pastikan logika pemilihan posisi `crop` benar-benar acak untuk menghindari penggunaan area degradasi yang sama berulang kali.
* **Kualitas Sumber:** Kualitas dan keragaman degradasi pada `sumber_degradasi.jpg` secara langsung menentukan kualitas dan keragaman dataset sintetis yang dihasilkan.