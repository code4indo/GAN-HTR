#!/usr/bin/env python3
"""
Batch Document Enhancement Tool
Untuk memproses banyak gambar sekaligus

Usage:
    python batch_enhancer.py --input_dir path/to/input/folder --output_dir path/to/output/folder
    python batch_enhancer.py --input_dir images/ --output_dir enhanced_images/ --model custom_model.h5
"""

import os
import sys
import argparse
import glob
from tqdm import tqdm

# Import our main enhancer
try:
    from document_enhancer import enhance_document
except ImportError:
    print("❌ Error: document_enhancer.py tidak ditemukan!")
    print("Pastikan file document_enhancer.py ada di folder yang sama.")
    sys.exit(1)

def batch_enhance_documents(input_dir, output_dir, model_path=None, supported_formats=None):
    """
    Batch processing untuk multiple documents
    
    Args:
        input_dir: Folder input dengan gambar-gambar
        output_dir: Folder output untuk hasil enhancement
        model_path: Path ke model weights
        supported_formats: List format yang didukung
    """
    
    if supported_formats is None:
        supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    
    print("🔍 GAN-HTR BATCH DOCUMENT ENHANCER")
    print("=" * 40)
    print(f"📂 Input Directory: {input_dir}")
    print(f"💾 Output Directory: {output_dir}")
    
    # Check input directory
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_files = []
    for fmt in supported_formats:
        pattern = os.path.join(input_dir, '**', fmt)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        print(f"Supported formats: {supported_formats}")
        return
    
    print(f"📷 Found {len(image_files)} images to process")
    
    # Process each image
    success_count = 0
    error_count = 0
    
    for img_path in tqdm(image_files, desc="Enhancing documents"):
        try:
            # Get relative path untuk maintain struktur folder
            rel_path = os.path.relpath(img_path, input_dir)
            
            # Create output path
            output_subdir = os.path.dirname(os.path.join(output_dir, rel_path))
            os.makedirs(output_subdir, exist_ok=True)
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(output_subdir, f"enhanced_{base_name}.png")
            
            # Skip jika sudah ada
            if os.path.exists(output_path):
                print(f"⏭️ Skipping (already exists): {rel_path}")
                continue
            
            # Enhance document
            enhance_document(
                input_path=img_path,
                output_path=output_path,
                model_path=model_path,
                show_comparison=False  # No plot untuk batch processing
            )
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            error_count += 1
    
    # Summary
    print(f"\n📊 BATCH PROCESSING COMPLETED")
    print(f"=" * 30)
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📁 Results saved in: {output_dir}")

def main():
    """Command line interface untuk batch processing"""
    parser = argparse.ArgumentParser(description='Batch GAN-HTR Document Enhancement Tool')
    parser.add_argument('--input_dir', '-i', required=True, help='Input directory dengan gambar-gambar')
    parser.add_argument('--output_dir', '-o', required=True, help='Output directory untuk hasil enhancement')
    parser.add_argument('--model', '-m', help='Path ke model weights (optional)')
    parser.add_argument('--formats', nargs='+', help='Supported image formats (default: jpg, png, tif, etc.)')
    
    args = parser.parse_args()
    
    try:
        batch_enhance_documents(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_path=args.model,
            supported_formats=args.formats
        )
        print("\n✅ Batch document enhancement completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
