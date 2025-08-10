import os
import random
import argparse
from PIL import Image, ImageFilter, ImageChops
from tqdm import tqdm

# --- KONFIGURASI ---
# Padding untuk memastikan crop lebih besar dari teks
PADDING_HORIZONTAL = 50
PADDING_VERTICAL = 30

# Pengaturan default untuk efek tinta adaptif
DEFAULT_DARKEN_FACTOR = 0.4  # 0.0=hitam, 1.0=tidak ada perubahan
DEFAULT_BLUR_RADIUS = 1.2
# --- AKHIR KONFIGURASI ---

def create_adaptive_synthetic_dataset(clean_dir, degradation_dir, output_dir, use_color, darken_factor, blur_radius):
    """
    Membuat dataset terdegradasi sintetis menggunakan teknik tinta adaptif
    di mana warna teks adalah versi lebih gelap dari background di bawahnya.
    """
    image_mode = "RGB" if use_color else "L"
    print("Memulai proses pembuatan dataset (Metode: Tinta Adaptif)...")
    print(f"  - Faktor Penggelapan Tinta: {darken_factor}")
    print(f"  - Radius Blur Tinta: {blur_radius}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Hasil akan disimpan di: {output_dir}")

    print(f"Memuat sumber degradasi dari: {degradation_dir}")
    try:
        degradation_sources_files = [f for f in os.listdir(degradation_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not degradation_sources_files:
            print(f"Error: Tidak ada file gambar yang ditemukan di {degradation_dir}")
            return
        degradation_images = [Image.open(os.path.join(degradation_dir, f)).convert(image_mode) for f in degradation_sources_files]
        print(f"Berhasil memuat {len(degradation_images)} gambar sumber degradasi.")
    except FileNotFoundError:
        print(f"Error: Direktori sumber degradasi tidak ditemukan di {degradation_dir}")
        return

    try:
        clean_image_files = [f for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not clean_image_files:
            print(f"Error: Tidak ada file gambar bersih yang ditemukan di {clean_dir}")
            return
        print(f"Ditemukan {len(clean_image_files)} gambar bersih untuk diproses.")
    except FileNotFoundError:
        print(f"Error: Direktori gambar bersih tidak ditemukan di {clean_dir}")
        return

    print("\nMemproses gambar...")
    processed_count = 0
    for filename in tqdm(clean_image_files, desc="Generating Adaptive Images"):
        clean_image_path = os.path.join(clean_dir, filename)
        
        with Image.open(clean_image_path) as clean_img:
            clean_img_l = clean_img.convert("L")
            text_mask_inv = ImageChops.invert(clean_img_l)
            
            text_width, text_height = text_mask_inv.size
            crop_width = text_width + PADDING_HORIZONTAL
            crop_height = text_height + PADDING_VERTICAL

            suitable_sources = [img for img in degradation_images if img.width >= crop_width and img.height >= crop_height]
            if not suitable_sources:
                continue

            source_img = random.choice(suitable_sources)
            source_width, source_height = source_img.size

            max_x = source_width - crop_width
            max_y = source_height - crop_height
            crop_x = random.randint(0, max_x)
            crop_y = random.randint(0, max_y)

            background_crop = source_img.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

            # --- LOGIKA TINTA ADAPTIF ---
            text_canvas_mask = Image.new("L", (crop_width, crop_height), 0)
            paste_x = random.randint(0, PADDING_HORIZONTAL)
            paste_y = random.randint(0, PADDING_VERTICAL)
            text_canvas_mask.paste(text_mask_inv, (paste_x, paste_y))

            ink_bleed_mask = text_canvas_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # 1. Buat layer tinta dengan menggelapkan background
            darken_value = int(255 * darken_factor)
            darken_color = (darken_value, darken_value, darken_value) if image_mode == "RGB" else darken_value
            ink_layer = ImageChops.multiply(background_crop, Image.new(image_mode, background_crop.size, darken_color))

            # 2. Gabungkan background dengan layer tinta menggunakan mask
            final_image = background_crop.copy()
            final_image.paste(ink_layer, (0, 0), ink_bleed_mask)
            # --- AKHIR LOGIKA ADAPTIF ---

            output_path = os.path.join(output_dir, filename)
            final_image.save(output_path)
            processed_count += 1

    print("\nProses selesai!")
    print(f"Total {processed_count} gambar terdegradasi telah dibuat di {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Membuat dataset terdegradasi sintetis menggunakan metode TINTA ADAPTIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--clean-dir", type=str, help="Direktori berisi gambar teks bersih.")
    parser.add_argument("--degradation-dir", type=str, help="Direktori berisi gambar sumber degradasi.")
    parser.add_argument("--output-dir", type=str, help="Direktori untuk menyimpan hasil gambar.")
    parser.add_argument("--color", action="store_true", help="Menghasilkan gambar berwarna (mode RGB).")
    
    parser.add_argument(
        "--darken-factor",
        type=float,
        default=DEFAULT_DARKEN_FACTOR,
        help="Faktor penggelapan tinta (0.0=hitam, 1.0=tidak berubah)."
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=DEFAULT_BLUR_RADIUS,
        help="Radius blur untuk simulasi efek tinta luntur."
    )
    
    args = parser.parse_args()
    
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    clean_dir = args.clean_dir or os.path.join(PROJECT_ROOT, "nan_raw/test/images")
    degradation_dir = args.degradation_dir or os.path.join(PROJECT_ROOT, "datasets/background_images_extracted")
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "datasets/synthetic_test_degraded_adaptive")

    create_adaptive_synthetic_dataset(
        clean_dir=clean_dir,
        degradation_dir=degradation_dir,
        output_dir=output_dir,
        use_color=args.color,
        darken_factor=args.darken_factor,
        blur_radius=args.blur_radius
    )
