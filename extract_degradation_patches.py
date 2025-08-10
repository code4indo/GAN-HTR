import os
import argparse
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
import re

# Menonaktifkan peringatan DecompressionBomb dari PIL untuk file besar
Image.MAX_IMAGE_PIXELS = None

def parse_patch_sizes(sizes_str):
    """Mengurai string ukuran patch (e.g., "256x256,1024x512") menjadi list of tuples."""
    sizes = []
    for size_pair in sizes_str.split(','):
        match = re.match(r'(\d+)x(\d+)', size_pair.strip())
        if match:
            sizes.append((int(match.group(1)), int(match.group(2))))
    if not sizes:
        raise ValueError("Format ukuran patch tidak valid. Gunakan format seperti '256x256,1024x512'.")
    # Urutkan dari yang terbesar ke terkecil untuk efisiensi
    sizes.sort(key=lambda x: x[0] * x[1], reverse=True)
    return sizes

def extract_patches(source_dir, output_dir, patch_sizes, std_threshold, text_threshold_percent, stride_factor):
    """
    Mengekstrak potongan gambar (patch) yang memiliki variansi tinggi (kerusakan) 
    dan sedikit konten teks dari dokumen sumber.
    """
    print(f"Memulai ekstraksi patch...")
    print(f"  - Direktori Sumber: {source_dir}")
    print(f"  - Direktori Output: {output_dir}")
    print(f"  - Ukuran Patch: {patch_sizes}")
    print(f"  - Min. Standar Deviasi (Kerusakan): {std_threshold}")
    print(f"  - Maks. Persentase Teks: {text_threshold_percent}%")
    print(f"  - Faktor Stride: {stride_factor}")

    os.makedirs(output_dir, exist_ok=True)
    
    source_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    if not source_files:
        print(f"Error: Tidak ada file gambar yang ditemukan di {source_dir}")
        return

    total_patches_saved = 0

    for filename in tqdm(source_files, desc="Memproses Dokumen Sumber"):
        source_path = os.path.join(source_dir, filename)
        try:
            with Image.open(source_path) as img:
                # Konversi ke grayscale untuk analisis
                img_gray = img.convert('L')
                
                for patch_width, patch_height in tqdm(patch_sizes, desc=f"  Ukuran Patch ({filename})", leave=False):
                    stride_w = int(patch_width * stride_factor)
                    stride_h = int(patch_height * stride_factor)

                    if img.width < patch_width or img.height < patch_height:
                        continue # Lewati jika gambar sumber lebih kecil dari patch

                    for y in range(0, img.height - patch_height + 1, stride_h):
                        for x in range(0, img.width - patch_width + 1, stride_w):
                            # Crop patch
                            patch = img_gray.crop((x, y, x + patch_width, y + patch_height))
                            
                            # 1. Analisis Kerusakan (Variansi)
                            patch_array = np.array(patch)
                            std_dev = np.std(patch_array)
                            
                            if std_dev < std_threshold:
                                continue # Lewati jika patch terlalu 'flat' atau tidak cukup rusak

                            # 2. Analisis Teks (Piksel Gelap)
                            # Anggap piksel di bawah 128 adalah kandidat teks
                            dark_pixels = np.sum(patch_array < 128)
                            text_percent = (dark_pixels / patch_array.size) * 100
                            
                            if text_percent > text_threshold_percent:
                                continue # Lewati jika terlalu banyak konten seperti teks

                            # 3. Simpan Patch
                            # Gunakan patch asli (berwarna) jika ada
                            patch_to_save = img.crop((x, y, x + patch_width, y + patch_height))
                            
                            # Buat nama file yang deskriptif
                            base_filename = os.path.splitext(filename)[0]
                            savename = f"{base_filename}_patch_{patch_width}x{patch_height}_x{x}_y{y}.png"
                            save_path = os.path.join(output_dir, savename)
                            
                            patch_to_save.save(save_path)
                            total_patches_saved += 1

        except Exception as e:
            print(f"Error memproses file {filename}: {e}")

    print(f"\nProses ekstraksi selesai.")
    print(f"Total {total_patches_saved} patch berhasil disimpan di {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ekstrak patch degradasi dari dokumen sumber.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="datasets/anriRusak",
        help="Direktori berisi dokumen sumber yang rusak."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/background_images_extracted",
        help="Direktori untuk menyimpan patch yang diekstrak."
    )
    parser.add_argument(
        "--patch-sizes",
        type=str,
        default="3000x1024,2048x512,1024x512,512x512,256x256",
        help="Daftar ukuran patch yang dipisahkan koma (e.g., '512x512,1024x512')."
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=15.0,
        help="Ambang batas minimum standar deviasi. Patch di bawah nilai ini dianggap terlalu 'flat' dan akan dilewati."
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=5.0,
        help="Ambang batas maksimum persentase piksel gelap (dianggap sebagai teks). Patch di atas nilai ini akan dilewati."
    )
    parser.add_argument(
        "--stride-factor",
        type=float,
        default=0.5,
        help="Faktor tumpang tindih (overlap) untuk sliding window. 0.5 berarti overlap 50%."
    )
    
    args = parser.parse_args()
    
    try:
        patch_sizes_list = parse_patch_sizes(args.patch_sizes)
        extract_patches(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            patch_sizes=patch_sizes_list,
            std_threshold=args.std_threshold,
            text_threshold_percent=args.text_threshold,
            stride_factor=args.stride_factor
        )
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
