import cv2
import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm
from augraphy import AugraphyPipeline, InkBleed, LowInkRandomLines, LowInkPeriodicLines, PaperFactory, ColorPaper, WaterMark, Stains, DirtyRollers, Folding, BleedThrough, BadPhotoCopy, LightingGradient, SubtleNoise, Jpeg, DirtyDrum

# 1. DEFINISIKAN PIPELINE AUGMENTASI
# Setiap fase mensimulasikan bagian dari siklus hidup dokumen.
def create_augraphy_pipeline(enabled_augmentations):
    # Fase Tinta: Mensimulasikan ketidaksempurnaan saat pencetakan.
    ink_phase = []
    if "InkBleed" in enabled_augmentations:
        ink_phase.append(InkBleed(p=enabled_augmentations["InkBleed"]))
    if "LowInkRandomLines" in enabled_augmentations:
        ink_phase.append(LowInkRandomLines(p=enabled_augmentations["LowInkRandomLines"], use_consistent_lines=False))
    if "LowInkPeriodicLines" in enabled_augmentations:
        ink_phase.append(LowInkPeriodicLines(p=enabled_augmentations["LowInkPeriodicLines"], use_consistent_lines=False))
    if "DirtyDrum" in enabled_augmentations:
        # Default severe parameters for DirtyDrum
        direction = enabled_augmentations.get("DirtyDrumDirection", 2) # Both horizontal and vertical
        line_width = enabled_augmentations.get("DirtyDrumLineWidth", (1, 4))
        line_concentration = enabled_augmentations.get("DirtyDrumLineConcentration", 0.3)
        noise_intensity = enabled_augmentations.get("DirtyDrumNoiseIntensity", 0.5)
        noise_intensity_value = enabled_augmentations.get("DirtyDrumNoiseIntensityValue", (0, 30))
        ink_phase.append(DirtyDrum(p=enabled_augmentations["DirtyDrum"],
                                   direction=direction,
                                   line_width_range=line_width,
                                   line_concentration=line_concentration,
                                   noise_intensity=noise_intensity,
                                   noise_value=noise_intensity_value))

    # Fase Kertas: Mensimulasikan kerusakan fisik pada kertas.
    paper_phase = []
    if "PaperFactory" in enabled_augmentations:
        paper_phase.append(PaperFactory(p=enabled_augmentations["PaperFactory"]))
    if "ColorPaper" in enabled_augmentations:
        paper_phase.append(ColorPaper(p=enabled_augmentations["ColorPaper"]))
    if "WaterMark" in enabled_augmentations:
        paper_phase.append(WaterMark(p=enabled_augmentations["WaterMark"]))
    if "Stains" in enabled_augmentations:
        paper_phase.append(Stains(p=enabled_augmentations["Stains"]))
    if "DirtyRollers" in enabled_augmentations:
        paper_phase.append(DirtyRollers(p=enabled_augmentations["DirtyRollers"]))
    if "Folding" in enabled_augmentations:
        paper_phase.append(Folding(p=enabled_augmentations["Folding"]))
    if "BleedThrough" in enabled_augmentations:
        # Check if custom intensity is provided, otherwise use a severe default
        if "BleedThroughIntensity" in enabled_augmentations:
            intensity_range = enabled_augmentations["BleedThroughIntensity"]
        else:
            intensity_range = (0.4, 0.8) # Default severe intensity
        
        # Check if extreme parameters are provided
        alpha = enabled_augmentations.get("BleedThroughAlpha", 0.1)  # Lower alpha = more visible bleed
        offsets = enabled_augmentations.get("BleedThroughOffsets", (30, 30))  # Larger offsets = more spread
        color_range = enabled_augmentations.get("BleedThroughColorRange", (0, 150))  # Darker colors
        ksize = enabled_augmentations.get("BleedThroughKsize", (25, 25))  # Larger kernel = more blur
        sigmaX = enabled_augmentations.get("BleedThroughSigmaX", 2)  # More blur
        
        paper_phase.append(BleedThrough(
            p=enabled_augmentations["BleedThrough"], 
            intensity_range=intensity_range,
            alpha=alpha,
            offsets=offsets,
            color_range=color_range,
            ksize=ksize,
            sigmaX=sigmaX
        ))

    # Fase Pasca-Produksi: Mensimulasikan artefak dari proses scanning/fotokopi.
    post_phase = []
    if "BadPhotoCopy" in enabled_augmentations:
        post_phase.append(BadPhotoCopy(p=enabled_augmentations["BadPhotoCopy"]))
    if "LightingGradient" in enabled_augmentations:
        post_phase.append(LightingGradient(p=enabled_augmentations["LightingGradient"]))
    if "SubtleNoise" in enabled_augmentations:
        post_phase.append(SubtleNoise(p=enabled_augmentations["SubtleNoise"]))
    if "Jpeg" in enabled_augmentations:
        post_phase.append(Jpeg(p=enabled_augmentations["Jpeg"]))

    # Gabungkan semua fase menjadi satu pipeline utama
    return AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase
    )

# 2. FUNGSI UTAMA UNTUK MEMPROSES GAMBAR
def process_images(dataset_split, enabled_augmentations, num_images=None):
    """
    Menerapkan pipeline Augraphy pada dataset yang ditentukan (train, valid, atau test).
    """
    print(f"🚀 Memulai proses degradasi untuk set: {dataset_split}")

    # Tentukan path input dan output
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_images_base_dir = os.path.join(base_dir, "datasets", "iam_raw")
    output_base_dir = os.path.join(base_dir, "datasets", "iam_augraphy_distorted")

    # Buat timestamp untuk folder output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_with_timestamp = os.path.join(output_base_dir, timestamp, dataset_split, "images")

    # Tentukan path berdasarkan dataset_split
    if dataset_split == 'valid':
        file_list_path = os.path.join(base_dir, "Sets", "list_valid_iam.txt")
        # IAM validation set is split into two parts, we use the first one
        raw_images_dir = os.path.join(raw_images_base_dir, "validation", "images")
    else: # train or test
        file_list_path = os.path.join(base_dir, "Sets", f"list_{dataset_split}_iam.txt")
        raw_images_dir = os.path.join(raw_images_base_dir, dataset_split, "images")

    output_dir = output_dir_with_timestamp
    os.makedirs(output_dir, exist_ok=True)

    # Baca daftar file
    try:
        with open(file_list_path, 'r') as f:
            file_list = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"❌ Error: File list tidak ditemukan di {file_list_path}")
        print("Pastikan Anda sudah menjalankan 'create_iam_file_lists.py' terlebih dahulu.")
        return

    # Apply num_images limit if specified
    if num_images is not None and num_images > 0:
        file_list = file_list[:num_images]
        print(f"Processing only the first {num_images} images.")

    # Inisialisasi pipeline
    pipeline = create_augraphy_pipeline(enabled_augmentations)

    print(f"Found {len(file_list)} images to process.")
    print(f"Gambar asli akan dibaca dari: {raw_images_dir}")
    print(f"Hasil degradasi akan disimpan di: {output_dir}")
    print(f"Augmentasi yang diaktifkan: {enabled_augmentations}")

    # Proses setiap gambar dengan progress bar
    for filename in tqdm(file_list, desc=f"Processing {dataset_split} set"):
        image_path = os.path.join(raw_images_dir, f"{filename}.png")
        output_path = os.path.join(output_dir, f"{filename}.png")

        if not os.path.exists(image_path):
            # Handle cases where image might be in a subdirectory (like in IAM train set)
            # Example path: a01/a01-000u/a01-000u-00.png
            parts = filename.split('-')
            if len(parts) > 1:
                sub_dir = parts[0]
                folder_name = f"{sub_dir}-{parts[1]}"
                image_path_alt = os.path.join(raw_images_dir, sub_dir, folder_name, f"{filename}.png")
                if os.path.exists(image_path_alt):
                    image_path = image_path_alt
                else:
                    # print(f"⚠️  Peringatan: Gambar tidak ditemukan: {image_path} atau {image_path_alt}")
                    continue
            else:
                # print(f"⚠️  Peringatan: Gambar tidak ditemukan: {image_path}")
                continue

        # Muat gambar
        image = cv2.imread(image_path)
        if image is None:
            # print(f"⚠️  Peringatan: Gagal memuat gambar: {image_path}")
            continue

        # Terapkan pipeline Augraphy
        degraded_image = pipeline(image)

        # Simpan hasilnya
        cv2.imwrite(output_path, degraded_image)

    print(f"✅ Proses degradasi untuk set '{dataset_split}' selesai.")
    print(f"Gambar tersimpan di: {output_dir}")

# 3. ENTRY POINT SCRIPT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic degraded document images using Augraphy.")
    parser.add_argument("dataset_split", type=str, choices=["train", "valid", "test"],
                        help="Dataset split to process (train, valid, or test).")

    # Add arguments for each augmentation with default probability 0.0 (meaning not included by default)
    # User can specify --InkBleed 0.7 to enable it with 70% probability
    parser.add_argument("--InkBleed", type=float, default=0.0, help="Probability for InkBleed augmentation (0.0-1.0).")
    parser.add_argument("--LowInkRandomLines", type=float, default=0.0, help="Probability for LowInkRandomLines augmentation (0.0-1.0).")
    parser.add_argument("--LowInkPeriodicLines", type=float, default=0.0, help="Probability for LowInkPeriodicLines augmentation (0.0-1.0).")
    parser.add_argument("--DirtyDrum", type=float, default=0.0, help="Probability for DirtyDrum augmentation (0.0-1.0).")
    parser.add_argument("--DirtyDrumDirection", type=int, default=2, choices=[0, 1, 2], help="Direction for DirtyDrum (0: horizontal, 1: vertical, 2: both). Default: 2.")
    parser.add_argument("--DirtyDrumLineWidthRange", type=int, nargs=2, default=[1, 4], help="Min and max line width for DirtyDrum. Default: 1 4.")
    parser.add_argument("--DirtyDrumLineConcentration", type=float, default=0.3, help="Line concentration for DirtyDrum (0.0-1.0). Default: 0.3.")
    parser.add_argument("--DirtyDrumNoiseIntensity", type=float, default=0.5, help="Noise intensity for DirtyDrum (0.0-1.0). Default: 0.5.")
    parser.add_argument("--DirtyDrumNoiseValue", type=int, nargs=2, default=[0, 30], help="Min and max noise intensity value for DirtyDrum. Default: 0 30.")
    parser.add_argument("--PaperFactory", type=float, default=0.0, help="Probability for PaperFactory augmentation (0.0-1.0).")
    parser.add_argument("--ColorPaper", type=float, default=0.0, help="Probability for ColorPaper augmentation (0.0-1.0).")
    parser.add_argument("--WaterMark", type=float, default=0.0, help="Probability for WaterMark augmentation (0.0-1.0).")
    parser.add_argument("--Stains", type=float, default=0.0, help="Probability for Stains augmentation (0.0-1.0).")
    parser.add_argument("--DirtyRollers", type=float, default=0.0, help="Probability for DirtyRollers augmentation (0.0-1.0).")
    parser.add_argument("--Folding", type=float, default=0.0, help="Probability for Folding augmentation (0.0-1.0).")
    parser.add_argument("--BleedThrough", type=float, default=0.0, help="Probability for BleedThrough augmentation (0.0-1.0).")
    parser.add_argument("--BleedThroughIntensity", type=float, nargs=2, default=[0.4, 0.8], help="Min and max intensity for BleedThrough augmentation (0.0-1.0). Default: 0.4 0.8 (more severe).")
    parser.add_argument("--BleedThroughAlpha", type=float, default=0.1, help="Alpha transparency for BleedThrough (0.0-1.0). Lower values = more visible bleed. Default: 0.1.")
    parser.add_argument("--BleedThroughOffsets", type=int, nargs=2, default=[30, 30], help="X and Y offset for BleedThrough effect. Default: 30 30.")
    parser.add_argument("--BleedThroughColorRange", type=int, nargs=2, default=[0, 150], help="Min and max color values for BleedThrough. Default: 0 150 (darker).")
    parser.add_argument("--BleedThroughKsize", type=int, nargs=2, default=[25, 25], help="Kernel size for BleedThrough blur. Default: 25 25.")
    parser.add_argument("--BleedThroughSigmaX", type=float, default=2.0, help="Sigma X for BleedThrough blur. Default: 2.0.")
    parser.add_argument("--BadPhotoCopy", type=float, default=0.0, help="Probability for BadPhotoCopy augmentation (0.0-1.0).")
    parser.add_argument("--LightingGradient", type=float, default=0.0, help="Probability for LightingGradient augmentation (0.0-1.0).")
    parser.add_argument("--SubtleNoise", type=float, default=0.0, help="Probability for SubtleNoise augmentation (0.0-1.0).")
    parser.add_argument("--Jpeg", type=float, default=0.0, help="Probability for Jpeg augmentation (0.0-1.0).")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to process. Process all if not specified.")

    args = parser.parse_args()

    dataset_split = args.dataset_split

    # Collect enabled augmentations and their probabilities/intensities
    enabled_augmentations = {}
    for aug_name in ["InkBleed", "LowInkRandomLines", "LowInkPeriodicLines", "DirtyDrum",
                     "PaperFactory", "ColorPaper", "WaterMark", "Stains", "DirtyRollers", "Folding", "BleedThrough",
                     "BadPhotoCopy", "LightingGradient", "SubtleNoise", "Jpeg"]:
        prob = getattr(args, aug_name)
        if prob > 0.0: # Only include if probability is greater than 0
            enabled_augmentations[aug_name] = prob

    # Handle BleedThrough parameters separately
    if args.BleedThroughIntensity is not None and args.BleedThroughIntensity != [0.4, 0.8]:
        enabled_augmentations["BleedThroughIntensity"] = tuple(args.BleedThroughIntensity)
    if args.BleedThroughAlpha is not None and args.BleedThroughAlpha != 0.1:
        enabled_augmentations["BleedThroughAlpha"] = args.BleedThroughAlpha
    if args.BleedThroughOffsets is not None and args.BleedThroughOffsets != [30, 30]:
        enabled_augmentations["BleedThroughOffsets"] = tuple(args.BleedThroughOffsets)
    if args.BleedThroughColorRange is not None and args.BleedThroughColorRange != [0, 150]:
        enabled_augmentations["BleedThroughColorRange"] = tuple(args.BleedThroughColorRange)
    if args.BleedThroughKsize is not None and args.BleedThroughKsize != [25, 25]:
        enabled_augmentations["BleedThroughKsize"] = tuple(args.BleedThroughKsize)
    if args.BleedThroughSigmaX is not None and args.BleedThroughSigmaX != 2.0:
        enabled_augmentations["BleedThroughSigmaX"] = args.BleedThroughSigmaX

    # Handle DirtyDrum parameters separately
    if args.DirtyDrum > 0.0: # Only if DirtyDrum is enabled
        if args.DirtyDrumDirection != 2:
            enabled_augmentations["DirtyDrumDirection"] = args.DirtyDrumDirection
        if args.DirtyDrumLineWidthRange != [1, 4]:
            enabled_augmentations["DirtyDrumLineWidth"] = tuple(args.DirtyDrumLineWidthRange)
        if args.DirtyDrumLineConcentration != 0.3:
            enabled_augmentations["DirtyDrumLineConcentration"] = args.DirtyDrumLineConcentration
        if args.DirtyDrumNoiseIntensity != 0.5:
            enabled_augmentations["DirtyDrumNoiseIntensity"] = args.DirtyDrumNoiseIntensity
        if args.DirtyDrumNoiseValue != [0, 30]:
            enabled_augmentations["DirtyDrumNoiseIntensityValue"] = tuple(args.DirtyDrumNoiseValue)

    # If no specific augmentations are provided, use a default set
    # Check if enabled_augmentations is empty or only contains intensity/parameter arguments
    if not enabled_augmentations or all(key.endswith("Intensity") or key.startswith("DirtyDrum") for key in enabled_augmentations.keys()):
        print("No specific augmentations provided or only intensity/parameter arguments. Using default pipeline.")
        enabled_augmentations = {
            "InkBleed": 0.4,
            "LowInkRandomLines": 0.5,
            "LowInkPeriodicLines": 0.5,
            "DirtyDrum": 0.6, # Default probability for DirtyDrum
            "DirtyDrumDirection": 2, # Default
            "DirtyDrumLineWidth": (1, 4), # Default
            "DirtyDrumLineConcentration": 0.3, # Default
            "DirtyDrumNoiseIntensity": 0.5, # Default
            "DirtyDrumNoiseIntensityValue": (0, 30), # Default
            "PaperFactory": 0.3,
            "ColorPaper": 0.3,
            "WaterMark": 0.4,
            "Stains": 0.3,
            "DirtyRollers": 0.5,
            "Folding": 0.5,
            "BleedThrough": 0.4, # Default probability for BleedThrough
            "BleedThroughIntensity": (0.4, 0.8), # Default severe intensity for BleedThrough
            "BleedThroughAlpha": 0.1, # Default alpha transparency
            "BleedThroughOffsets": (30, 30), # Default offsets
            "BleedThroughColorRange": (0, 150), # Default color range
            "BleedThroughKsize": (25, 25), # Default kernel size
            "BleedThroughSigmaX": 2.0, # Default sigma
            "BadPhotoCopy": 0.5,
            "LightingGradient": 0.4,
            "SubtleNoise": 0.5,
            "Jpeg": 0.4,
        }

    process_images(dataset_split, enabled_augmentations, args.num_images)