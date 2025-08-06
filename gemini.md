
# Proses Pengembangan Model Berdasarkan Paper "Enhance to read better"

Berikut adalah ringkasan tahapan dari persiapan data hingga pelatihan model yang dijelaskan dalam paper "Enhance to read better: A Multi-Task Adversarial Network for Handwritten Document Image Enhancement".

## 1. Persiapan Data

Tantangan utama adalah dataset binerisasi dokumen umumnya tidak memiliki informasi teks (ground truth), yang krusial untuk melatih komponen *recognizer*. Oleh karena itu, prosesnya adalah membuat dataset terdegradasi secara sintetis.

1.  **Pemilihan Dataset Dasar:** Menggunakan dataset standar untuk Handwritten Text Recognition (HTR) yang sudah memiliki pasangan gambar bersih dan transkripsi teksnya.
    *   **KHATT:** Untuk teks Arab.
    *   **IAM:** Untuk teks Latin.

2.  **Pembuatan Gambar Terdegradasi (Synthetic Degradation):** Gambar-gambar bersih dari dataset dasar diubah secara artifisial untuk mensimulasikan kerusakan pada dokumen nyata. Proses ini meliputi:
    *   **Menambahkan Latar Belakang (Background):** Menempelkan gambar teks pada gambar latar belakang yang mengandung artefak/noise. Latar belakang ini diambil dari dokumen-dokumen historis sungguhan (misalnya, Nabuco, Bickley diary).
    *   **Menerapkan Distorsi:** Mengaplikasikan berbagai jenis distorsi secara acak, seperti:
        *   Dilasi dan Erosi dengan ukuran kernel yang acak.
        *   Blur dengan tingkat yang bervariasi.
        *   Menambahkan garis-garis vertikal acak untuk mensimulasikan goresan atau lipatan.

3.  **Hasil Akhir Data:** Setiap set data kini memiliki tiga komponen:
    *   Gambar asli yang terdegradasi (input).
    *   Gambar versi bersih (target visual / ground truth).
    *   Transkripsi teks yang benar (target keterbacaan / ground truth).

## 2. Arsitektur dan Pelatihan Model

Model yang diusulkan adalah sebuah Generative Adversarial Network (GAN) yang dimodifikasi untuk tujuan ganda: membersihkan gambar dan memastikan keterbacaannya.

1.  **Komponen Model:**
    *   **Generator:** Sebuah arsitektur U-Net (sejenis auto-encoder) yang bertugas mengambil gambar terdegradasi dan menghasilkan versi yang bersih.
    *   **Discriminator:** Sebuah jaringan CNN yang bertugas membedakan antara gambar yang "asli" bersih (dari dataset) dan gambar yang "palsu" (dihasilkan oleh Generator). Tujuannya adalah memaksa Generator menghasilkan gambar yang realistis secara visual.
    *   **Handwritten Text Recognizer (HTR):** Sebuah model CRNN (Convolutional Recurrent Neural Network) yang bertugas membaca teks dari gambar yang dihasilkan Generator. Tujuannya adalah memastikan teks pada gambar yang sudah dibersihkan tetap utuh dan dapat dibaca.

2.  **Proses Pelatihan (Training):**
    *   Ketiga komponen dilatih secara bersamaan (*jointly*).
    *   **Generator** dioptimalkan berdasarkan tiga sinyal *loss* (kesalahan):
        1.  **Adversarial Loss:** Dari Discriminator, untuk membuat gambar tampak nyata.
        2.  **Content Loss (BCE Loss):** Perbedaan piksel-demi-piksel antara gambar yang dihasilkan dan gambar asli yang bersih, untuk memastikan kesamaan visual.
        3.  **Readability Loss (CTC Loss):** Dari HTR, untuk memastikan teks pada gambar yang dihasilkan sesuai dengan transkripsi ground truth.
    *   **Discriminator** dilatih untuk menjadi semakin pintar dalam membedakan gambar asli dan palsu.
    *   **HTR** dilatih untuk mengenali teks, baik dari gambar asli yang bersih maupun dari gambar yang dihasilkan Generator.

Dengan melatih ketiga komponen ini secara bersamaan, Generator belajar untuk tidak hanya "membersihkan" noise, tetapi juga "mempertahankan" goresan tinta yang merupakan bagian dari tulisan, sehingga menghasilkan gambar yang bersih sekaligus dapat dibaca dengan akurat oleh sistem HTR.
