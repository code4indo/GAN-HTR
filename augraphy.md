# Strategi Membuat Gambar Dokumen Terdegradasi dengan Augraphy

Dokumen ini menjelaskan strategi modern dan efektif untuk membuat dataset gambar dokumen terdegradasi secara sintetis menggunakan **Augraphy**, sebuah library Python yang canggih.

Pendekatan ini bertujuan untuk menghasilkan degradasi yang jauh lebih realistis dibandingkan dengan menerapkan filter-filter individual (seperti blur, erosi, dll.) secara terpisah.

## Apa itu Augraphy?

Augraphy adalah library augmentasi gambar yang dirancang khusus untuk mensimulasikan proses degradasi yang terjadi pada dokumen di dunia nyata. Library ini memungkinkan kita untuk meniru seluruh "siklus hidup" sebuah dokumen, mulai dari proses pencetakan hingga digitalisasi.

## Konsep Inti: Pipeline Augmentasi

Konsep utama dalam Augraphy adalah **Pipeline**. Sebuah pipeline adalah serangkaian langkah augmentasi yang akan dilewati oleh setiap gambar. Setiap langkah memiliki probabilitas untuk dijalankan, sehingga menghasilkan variasi output yang sangat kaya dan tidak terduga.

Pipeline ini secara cerdas meniru tiga fase utama dalam siklus hidup dokumen:

1.  **Fase Tinta (`Ink Phase`):** Mensimulasikan ketidaksempurnaan saat teks dicetak di atas kertas.
2.  **Fase Kertas (`Paper Phase`):** Mensimulasikan kerusakan fisik yang dialami dokumen setelah dicetak (misalnya, terlipat, kotor, basah, sobek).
3.  **Fase Pasca-Produksi (`Post Phase`):** Mensimulasikan artefak dan masalah yang muncul saat dokumen fisik di-scan atau difoto.

---

## Strategi Implementasi

Berikut adalah strategi langkah-demi-langkah untuk mengimplementasikan Augraphy dalam proyek ini.

### Langkah 1: Mulai dengan Gambar Bersih

Dasar dari proses ini adalah gambar dokumen yang bersih dan berkualitas tinggi, seperti yang tersedia di dataset IAM (`datasets/iam_raw/...`).

### Langkah 2: Definisikan Pipeline Augraphy

Langkah paling penting adalah mendefinisikan sebuah `AugraphyPipeline`. Di sinilah kita menentukan jenis-jenis degradasi yang ingin kita terapkan. Pipeline ini bisa berisi puluhan jenis augmentasi, masing-masing dengan probabilitasnya sendiri.

#### Fase 1: Simulasi Kertas dan Tinta (Efek Awal)

-   `PaperFactory`: Mengganti latar belakang putih bersih dengan tekstur kertas yang realistis, lengkap dengan serat-serat halus.
-   `InkBleed`: Mensimulasikan tinta yang sedikit "meleber" atau menyebar ke serat kertas, membuat pinggiran huruf tidak terlalu tajam.
-   `LowInkRandomLines` atau `LowInkPeriodicLines`: Meniru efek printer yang kehabisan tinta, yang menghasilkan garis-garis putih horizontal pada teks.

#### Fase 2: Tambahkan Degradasi Fisik (Kerusakan)

-   `Folding` atau `BookBinding`: Menambahkan efek lipatan, kerutan, atau bayangan seperti pada buku yang dijilid, memberikan kesan tiga dimensi.
-   `WaterMark` atau `Stains`: Menambahkan noda air, kopi, atau jenis noda lainnya secara realistis.
-   `DirtyRollers`: Mensimulasikan jejak kotoran vertikal dari roller mesin fotokopi.
-   `BleedThrough`: Membuat teks dari halaman belakang tampak samar-samar di halaman depan, seperti yang sering terjadi pada kertas tipis.

#### Fase 3: Simulasi Proses Digitalisasi (Artefak Digital)

-   `BadPhotoCopy`: Sebuah augmentasi gabungan yang meniru hasil fotokopi berkualitas buruk dengan noise, blur, dan kontras yang tidak merata.
-   `ScannerJitter`: Sedikit menggeser baris-baris piksel secara acak untuk mensimulasikan guncangan pada mesin scanner.
-   `LightingGradient`: Menciptakan efek pencahayaan yang tidak merata di seluruh permukaan gambar.
-   `Binarization`: Mengubah gambar menjadi hitam-putih (biner) menggunakan berbagai metode, yang seringkali merupakan langkah pre-processing di banyak sistem HTR dan dapat menghasilkan artefak.

### Langkah 3: Jalankan Gambar Melalui Pipeline

Setiap gambar bersih dari dataset akan dimasukkan ke dalam pipeline ini. Karena setiap augmentasi memiliki probabilitasnya sendiri (misalnya, `p=0.5` berarti 50% kemungkinan untuk dijalankan), setiap gambar output akan menjadi unik dan berbeda dari yang lain.

---

### Contoh Kode Pipeline Augraphy

Berikut adalah contoh skrip sederhana yang mendefinisikan dan menjalankan pipeline Augraphy.

```python
import cv2
from augraphy import *

# 1. Definisikan setiap fase augmentasi
ink_phase = [
    InkBleed(p=0.5),
    LowInkRandomLines(p=0.5),
]

paper_phase = [
    PaperFactory(p=0.5),
    WaterMark(p=0.3),
    Folding(p=0.5),
    DirtyRollers(p=0.4),
]

post_phase = [
    BadPhotoCopy(p=0.4),
    ScannerJitter(p=0.3),
    LightingGradient(p=0.5),
]

# 2. Gabungkan semua fase menjadi satu pipeline utama
pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)

# 3. Muat gambar bersih (contoh)
image = cv2.imread("path/to/clean_image.png")

# 4. Jalankan gambar melalui pipeline untuk menghasilkan degradasi
degraded_image = pipeline(image)

# 5. Simpan hasilnya
cv2.imwrite("path/to/degraded_image.png", degraded_image)

print("Gambar terdegradasi dengan Augraphy berhasil dibuat!")
```

---

## Keuntungan Menggunakan Strategi Augraphy

1.  **Realistis**: Hasil degradasi sangat mirip dengan dokumen yang rusak secara alami di dunia nyata.
2.  **Kompleks**: Mampu menciptakan kombinasi degradasi berlapis yang sulit dicapai dengan metode manual.
3.  **Terkontrol & Fleksibel**: Sangat mudah untuk menambah, menghapus, atau mengubah parameter setiap jenis degradasi dalam pipeline.
4.  **Efisiensi**: Menggantikan puluhan fungsi manual dengan satu pipeline yang terstruktur dan mudah dikelola.

Untuk mengintegrasikan pendekatan ini ke dalam proyek, kita dapat membuat skrip baru (misalnya, `distort_image_augraphy.py`) yang berisi pipeline di atas, lalu menjalankannya untuk setiap set data (`train`, `valid`, `test`).
