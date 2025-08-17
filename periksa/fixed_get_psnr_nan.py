#!/usr/bin/env python3
"""
Fixed version of get_psnr_iam function for NAN dataset
"""

import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def read_file(list_file_path):
    """Read file list"""
    with open(list_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))

def get_psnr_nan(scenario='S_iam_OP_debug', epoch_num=None):
    """
    Fixed PSNR calculation for NAN dataset
    
    Args:
        scenario: Training scenario name
        epoch_num: Specific epoch number to evaluate (if None, use latest)
    """
    rootPath = './'
    
    # Path untuk ground truth images (NAN dataset) - deteksi otomatis dari split yang sesuai
    gt_splits_to_try = ['test', 'validation', 'train']
    DatabasePathGT = None
    
    for split in gt_splits_to_try:
        potential_path = f'datasets/nan_raw_biner/{split}/images/'
        if os.path.exists(potential_path):
            DatabasePathGT = potential_path
            break
    
    if not DatabasePathGT:
        print("Error: Tidak ada direktori ground truth yang ditemukan!")
        return None
    
    # Path untuk enhanced images hasil GAN - coba beberapa variasi nama
    result_dir_patterns = [
        f'Result{scenario}',
        f'ResultGan{scenario}', 
        f'ResultGanS_{scenario}',
        f'Result_{scenario}'
    ]
    
    enhanced_base_path = None
    
    for result_pattern in result_dir_patterns:
        if os.path.exists(result_pattern):
            if epoch_num is not None:
                enhanced_base_path = f'{result_pattern}/epoch{epoch_num}/'
                if os.path.exists(enhanced_base_path):
                    break
            else:
                # Find latest epoch directory
                epoch_dirs = [d for d in os.listdir(result_pattern) if d.startswith('epoch')]
                if epoch_dirs:
                    latest_epoch = max(epoch_dirs, key=lambda x: int(x.replace('epoch', '')))
                    enhanced_base_path = f'{result_pattern}/{latest_epoch}/'
                    break
    
    if enhanced_base_path is None:
        print("Error: No result directory found!")
        print("Mencari direktori dengan pola:")
        for pattern in result_dir_patterns:
            exists = "✅" if os.path.exists(pattern) else "❌"
            print(f"  {exists} {pattern}")
        return None
    
    count_image = 1
    total_psnr = 0
    valid_images = 0
    
    # Try different file lists to find intersection with enhanced images
    list_files_to_try = [
        'Sets/list_test_nan.txt',
        'Sets/list_train_nan.txt', 
        'Sets/list_valid_nan.txt'
    ]
    
    list_image = []
    used_list_file = None
    
    for list_file in list_files_to_try:
        if os.path.exists(rootPath + list_file):
            temp_list = read_file(rootPath + list_file)
            
            # Cari intersection antara list dan enhanced images yang tersedia
            if os.path.exists(enhanced_base_path):
                enhanced_files = [f for f in os.listdir(enhanced_base_path) if f.endswith('.png')]
                enhanced_base_names = []
                
                for enhanced_file in enhanced_files:
                    # Remove .jpg.png or .png extension
                    if enhanced_file.endswith('.jpg.png'):
                        base_name = enhanced_file[:-8]  # Remove .jpg.png
                    elif enhanced_file.endswith('.png'):
                        base_name = enhanced_file[:-4]  # Remove .png
                    else:
                        continue
                    
                    # Remove numeric prefix (e.g., "173_" from "173_NL-HaNA_1.04.02_3660_0192.tif_r3l14")
                    if '_' in base_name and base_name.split('_')[0].isdigit():
                        base_name = '_'.join(base_name.split('_')[1:])
                    
                    enhanced_base_names.append(base_name)
                
                # Find intersection - bandingkan dengan dan tanpa prefix
                available_images = []
                for img in temp_list:
                    # Remove numeric prefix from list if exists
                    img_clean = img
                    if '_' in img and img.split('_')[0].isdigit():
                        img_clean = '_'.join(img.split('_')[1:])
                    
                    # Check if any enhanced image matches (with or without prefix)
                    for enhanced_base in enhanced_base_names:
                        enhanced_clean = enhanced_base
                        if '_' in enhanced_base and enhanced_base.split('_')[0].isdigit():
                            enhanced_clean = '_'.join(enhanced_base.split('_')[1:])
                        
                        if img_clean == enhanced_clean or img == enhanced_base:
                            available_images.append(img)
                            break
                
                if available_images:
                    list_image = available_images[:50]  # Limit untuk testing
                    used_list_file = list_file
                    print(f"✅ Menggunakan {list_file}: {len(available_images)} gambar cocok (menggunakan {len(list_image)} untuk evaluasi)")
                    break
                else:
                    print(f"⚠️  {list_file}: Tidak ada intersection dengan enhanced images")
    
    if not list_image:
        print(f"❌ Tidak ada gambar yang tersedia di enhanced directory!")
        print(f"   Enhanced images tersedia: {len(enhanced_base_names) if 'enhanced_base_names' in locals() else 0} files")
        if 'enhanced_base_names' in locals():
            print(f"   Contoh enhanced: {enhanced_base_names[:3]}")
        return None
    
    print(f"🔍 Calculating PSNR for {len(list_image)} images...")
    print(f"📁 GT Path: {DatabasePathGT}")
    print(f"📁 Enhanced Path: {enhanced_base_path}")
    
    for i, im in enumerate(list_image):
        try:
            # Ground truth image path - coba di berbagai split
            gt_image_path = None
            for split in ['test', 'validation', 'train']:
                potential_gt_path = f'datasets/nan_raw_biner/{split}/images/'
                if os.path.exists(potential_gt_path):
                    for ext in ['.jpg', '.png']:
                        test_path = os.path.join(potential_gt_path, im + ext)
                        if os.path.exists(test_path):
                            gt_image_path = test_path
                            break
                    if gt_image_path:
                        break
            
            if not gt_image_path:
                print(f"⚠️  GT image not found: {im}")
                continue
            
            # Enhanced image path - cari file dengan nama yang sesuai
            enhanced_image_path = None
            
            # Karena enhanced images memiliki ekstensi .jpg.png
            expected_enhanced_filename = im + ".jpg.png"
            enhanced_image_path = os.path.join(enhanced_base_path, expected_enhanced_filename)
            
            if not os.path.exists(enhanced_image_path):
                print(f"⚠️  Enhanced image not found: {expected_enhanced_filename}")
                continue
            
            # Load and process ground truth image
            gt_image = Image.open(gt_image_path)
            gt_image = gt_image.resize((1024, 128), Image.Resampling.LANCZOS)
            gt_image = gt_image.convert('L')  # Convert to grayscale
            gt_array = np.array(gt_image) / 255.0  # Normalize to [0,1]
            
            # Load and process enhanced image
            enhanced_image = Image.open(enhanced_image_path)
            enhanced_image = enhanced_image.resize((1024, 128), Image.Resampling.LANCZOS)
            enhanced_image = enhanced_image.convert('L')  # Convert to grayscale
            enhanced_array = np.array(enhanced_image) / 255.0  # Normalize to [0,1]
            
            # Calculate PSNR
            psnr_value = psnr(enhanced_array, gt_array)
            print(f"📊 Image {i+1:3d}: {im[:50]:50s} PSNR: {psnr_value:.2f}")
            
            total_psnr += psnr_value
            valid_images += 1
            
        except Exception as e:
            print(f"❌ Error processing {im}: {str(e)}")
            continue
    
    if valid_images > 0:
        average_psnr = total_psnr / valid_images
        print(f"\n📈 Results Summary:")
        print(f"   Total images processed: {valid_images}/{len(list_image)}")
        print(f"   Average PSNR: {average_psnr:.2f} dB")
        return average_psnr
    else:
        print("❌ No valid images found for PSNR calculation!")
        return None

if __name__ == '__main__':
    # Test the function
    result = get_psnr_nan('S_iam_OP_debug')
    if result:
        print(f"\n✅ PSNR calculation completed: {result:.2f} dB")
    else:
        print("\n❌ PSNR calculation failed!")
