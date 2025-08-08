import os
import random
import argparse
from PIL import Image, ImageChops
from tqdm import tqdm

# --- KONFIGURASI ---
# Path absolut ke direktori root proyek
PROJECT_ROOT = "/home/lambda_one/tesis/GAN-HTR"

# Direktori input
CLEAN_IMAGE_DIR = os.path.join(PROJECT_ROOT, "datasets/iam_raw/test/images")
DEGRADATION_SOURCE_DIR = os.path.join(PROJECT_ROOT, "datasets/cropDoc")

# Direktori output
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets/synthetic_iam_test_degraded")

# Padding untuk memastikan crop lebih besar dari teks
PADDING_HORIZONTAL = 50  # piksel
PADDING_VERTICAL = 30    # piksel

# --- AKHIR KONFIGURASI ---

def create_synthetic_dataset(use_color):
    """
    Membuat dataset terdegradasi sintetis dengan menggabungkan gambar teks bersih
    dengan potongan acak dari sumber gambar degradasi.
    """
    image_mode = "RGB" if use_color else "L"
    print(f"Memulai proses pembuatan dataset (Mode: {image_mode})...")

    # 1. Persiapan: Buat direktori output jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Hasil akan disimpan di: {OUTPUT_DIR}")

    # 2. Muat semua sumber degradasi ke memori
    print(f"Memuat sumber degradasi dari: {DEGRADATION_SOURCE_DIR}")
    try:
        degradation_sources_files = [f for f in os.listdir(DEGRADATION_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not degradation_sources_files:
            print(f"Error: Tidak ada file gambar yang ditemukan di {DEGRADATION_SOURCE_DIR}")
            return

        degradation_images = [Image.open(os.path.join(DEGRADATION_SOURCE_DIR, f)).convert(image_mode) for f in degradation_sources_files]
        print(f"Berhasil memuat {len(degradation_images)} gambar sumber degradasi dalam mode {image_mode}.")
    except FileNotFoundError:
        print(f"Error: Direktori sumber degradasi tidak ditemukan di {DEGRADATION_SOURCE_DIR}")
        return

    # 3. Dapatkan daftar gambar bersih
    try:
        clean_image_files = [f for f in os.listdir(CLEAN_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not clean_image_files:
            print(f"Error: Tidak ada file gambar bersih yang ditemukan di {CLEAN_IMAGE_DIR}")
            return
        print(f"Ditemukan {len(clean_image_files)} gambar bersih untuk diproses.")
    except FileNotFoundError:
        print(f"Error: Direktori gambar bersih tidak ditemukan di {CLEAN_IMAGE_DIR}")
        return

    # 4. Proses setiap gambar bersih
    print("\nMemproses gambar...")
    for filename in tqdm(clean_image_files, desc="Generating Degraded Images"):
        clean_image_path = os.path.join(CLEAN_IMAGE_DIR, filename)
        
        with Image.open(clean_image_path) as clean_img:
            # Gambar teks selalu dikonversi ke Grayscale ("L") untuk mendapatkan mask
            clean_img_l = clean_img.convert("L")
            text_width, text_height = clean_img_l.size

            # Tentukan dimensi crop
            crop_width = text_width + PADDING_HORIZONTAL
            crop_height = text_height + PADDING_VERTICAL

            # Pilih sumber degradasi secara acak
            source_img = random.choice(degradation_images)
            source_width, source_height = source_img.size

            # Pastikan crop tidak keluar dari batas sumber
            if crop_width > source_width or crop_height > source_height:
                # print(f"Warning: Melewati {filename} karena ukurannya lebih besar dari sumber degradasi.")
                continue

            # Pilih koordinat crop secara acak
            max_x = source_width - crop_width
            max_y = source_height - crop_height
            crop_x = random.randint(0, max_x)
            crop_y = random.randint(0, max_y)

            # Lakukan crop untuk mendapatkan latar belakang
            background_crop = source_img.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

            # Buat kanvas putih seukuran background_crop untuk menempatkan teks
            # Warna putih disesuaikan dengan mode gambar
            white_color = (255, 255, 255) if image_mode == "RGB" else 255
            text_canvas = Image.new(image_mode, (crop_width, crop_height), white_color)

            # Tentukan posisi acak untuk menempelkan teks (jitter)
            paste_x = random.randint(0, PADDING_HORIZONTAL)
            paste_y = random.randint(0, PADDING_VERTICAL)
            # Tempelkan gambar teks (mode L) ke kanvas (bisa L atau RGB)
            text_canvas.paste(clean_img_l, (paste_x, paste_y))

            # Gabungkan menggunakan mode "Multiply" untuk efek realistis
            final_image = ImageChops.multiply(background_crop, text_canvas)

            # Simpan hasil
            output_path = os.path.join(OUTPUT_DIR, filename)
            final_image.save(output_path)

    print("\nProses selesai!")
    print(f"Total {len(os.listdir(OUTPUT_DIR))} gambar terdegradasi telah dibuat di {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Membuat dataset terdegradasi sintetis dari gambar bersih dan sumber degradasi.")
    parser.add_argument(
        "--color",
        action="store_true",
        help="Gunakan flag ini untuk mempertahankan warna asli dari sumber degradasi (mode RGB). Defaultnya adalah Grayscale (mode L)."
    )
    args = parser.parse_args()
    
    create_synthetic_dataset(use_color=args.color)
