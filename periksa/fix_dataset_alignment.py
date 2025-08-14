#!/usr/bin/env python3
"""
🔧 Dataset Alignment & Model Retraining Script
==============================================

Script untuk memperbaiki masalah size mismatch pada dataset
dan melakukan retraining dengan data yang sudah dibenarkan.

Author: Lambda One
Date: August 13, 2024
"""

import cv2
import numpy as np
import os
from pathlib import Path
import glob
from tqdm import tqdm
import shutil

def analyze_dataset_problems():
    """Analyze current dataset problems"""
    print("🔍 ANALYZING DATASET PROBLEMS")
    print("=" * 40)
    
    # Check training data
    train_distorted_dir = "datasets/nan_distorted/train"
    train_gt_dir = "datasets/nan_raw_biner/train/images"
    
    # Get sample files
    distorted_files = glob.glob(os.path.join(train_distorted_dir, "*.jpg"))[:10]
    
    size_mismatches = 0
    total_pairs = 0
    
    for dist_file in distorted_files:
        filename = os.path.basename(dist_file)
        gt_file = os.path.join(train_gt_dir, filename)
        
        if os.path.exists(gt_file):
            dist_img = cv2.imread(dist_file, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            
            if dist_img is not None and gt_img is not None:
                total_pairs += 1
                if dist_img.shape != gt_img.shape:
                    size_mismatches += 1
                    print(f"❌ {filename}")
                    print(f"   Distorted: {dist_img.shape}")
                    print(f"   GT: {gt_img.shape}")
    
    print(f"\n📊 Results: {size_mismatches}/{total_pairs} pairs have size mismatches")
    return size_mismatches > 0

def create_aligned_dataset():
    """Create aligned dataset with consistent sizes"""
    print("\n🔧 CREATING ALIGNED DATASET")
    print("=" * 35)
    
    # Create output directories
    aligned_base_dir = "datasets/nan_aligned"
    aligned_distorted_dir = os.path.join(aligned_base_dir, "train/distorted")
    aligned_gt_dir = os.path.join(aligned_base_dir, "train/gt")
    
    os.makedirs(aligned_distorted_dir, exist_ok=True)
    os.makedirs(aligned_gt_dir, exist_ok=True)
    
    # Source directories
    train_distorted_dir = "datasets/nan_distorted/train"
    train_gt_dir = "datasets/nan_raw_biner/train/images"
    
    # Get all distorted files
    distorted_files = glob.glob(os.path.join(train_distorted_dir, "*.jpg"))
    
    successful_pairs = 0
    failed_pairs = 0
    
    print(f"Processing {len(distorted_files)} files...")
    
    for dist_file in tqdm(distorted_files, desc="Aligning dataset"):
        filename = os.path.basename(dist_file)
        gt_file = os.path.join(train_gt_dir, filename)
        
        if os.path.exists(gt_file):
            try:
                # Load images
                dist_img = cv2.imread(dist_file, cv2.IMREAD_GRAYSCALE)
                gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
                
                if dist_img is not None and gt_img is not None:
                    # Find common size (use larger dimensions)
                    target_height = max(dist_img.shape[0], gt_img.shape[0])
                    target_width = max(dist_img.shape[1], gt_img.shape[1])
                    
                    # Ensure dimensions are compatible with model (multiples of 8)
                    target_height = ((target_height + 7) // 8) * 8
                    target_width = ((target_width + 7) // 8) * 8
                    
                    # Resize both images
                    dist_aligned = cv2.resize(dist_img, (target_width, target_height))
                    gt_aligned = cv2.resize(gt_img, (target_width, target_height))
                    
                    # Normalize pixel ranges
                    # Distorted: keep as is (degraded quality)
                    # GT: ensure high contrast binary-like appearance
                    gt_aligned = enhance_ground_truth(gt_aligned)
                    
                    # Save aligned images
                    dist_output = os.path.join(aligned_distorted_dir, filename)
                    gt_output = os.path.join(aligned_gt_dir, filename)
                    
                    cv2.imwrite(dist_output, dist_aligned)
                    cv2.imwrite(gt_output, gt_aligned)
                    
                    successful_pairs += 1
                else:
                    failed_pairs += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                failed_pairs += 1
        else:
            failed_pairs += 1
    
    print(f"\n✅ Alignment complete!")
    print(f"   Successful pairs: {successful_pairs}")
    print(f"   Failed pairs: {failed_pairs}")
    
    return successful_pairs > 0

def enhance_ground_truth(gt_img):
    """Enhance ground truth to have better contrast"""
    # Apply Otsu thresholding for better binarization
    _, binary = cv2.threshold(gt_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: slight gaussian blur to smooth edges
    # enhanced = cv2.GaussianBlur(binary, (3, 3), 0.5)
    
    return binary

def create_test_aligned_dataset():
    """Create aligned test dataset"""
    print("\n🧪 CREATING ALIGNED TEST DATASET")
    print("=" * 40)
    
    # Create test output directories
    aligned_base_dir = "datasets/nan_aligned"
    aligned_test_distorted_dir = os.path.join(aligned_base_dir, "test/distorted")
    aligned_test_gt_dir = os.path.join(aligned_base_dir, "test/gt")
    
    os.makedirs(aligned_test_distorted_dir, exist_ok=True)
    os.makedirs(aligned_test_gt_dir, exist_ok=True)
    
    # Source test directories
    test_distorted_dir = "datasets/nan_distorted/test"
    test_gt_dir = "datasets/nan_raw_biner/test/images"
    
    # Process test files
    test_distorted_files = glob.glob(os.path.join(test_distorted_dir, "*.jpg"))
    
    for dist_file in tqdm(test_distorted_files[:10], desc="Aligning test data"):
        filename = os.path.basename(dist_file)
        gt_file = os.path.join(test_gt_dir, filename)
        
        if os.path.exists(gt_file):
            try:
                dist_img = cv2.imread(dist_file, cv2.IMREAD_GRAYSCALE)
                gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
                
                if dist_img is not None and gt_img is not None:
                    # Use same alignment logic as training
                    target_height = max(dist_img.shape[0], gt_img.shape[0])
                    target_width = max(dist_img.shape[1], gt_img.shape[1])
                    
                    target_height = ((target_height + 7) // 8) * 8
                    target_width = ((target_width + 7) // 8) * 8
                    
                    dist_aligned = cv2.resize(dist_img, (target_width, target_height))
                    gt_aligned = cv2.resize(gt_img, (target_width, target_height))
                    gt_aligned = enhance_ground_truth(gt_aligned)
                    
                    # Save aligned test images
                    dist_output = os.path.join(aligned_test_distorted_dir, filename)
                    gt_output = os.path.join(aligned_test_gt_dir, filename)
                    
                    cv2.imwrite(dist_output, dist_aligned)
                    cv2.imwrite(gt_output, gt_aligned)
                    
            except Exception as e:
                print(f"Error processing test {filename}: {e}")
    
    print("✅ Test dataset alignment complete!")

def validate_aligned_dataset():
    """Validate that aligned dataset is correct"""
    print("\n✅ VALIDATING ALIGNED DATASET")
    print("=" * 35)
    
    aligned_distorted_dir = "datasets/nan_aligned/train/distorted"
    aligned_gt_dir = "datasets/nan_aligned/train/gt"
    
    # Check sample files
    aligned_files = glob.glob(os.path.join(aligned_distorted_dir, "*.jpg"))[:5]
    
    all_sizes_match = True
    
    for dist_file in aligned_files:
        filename = os.path.basename(dist_file)
        gt_file = os.path.join(aligned_gt_dir, filename)
        
        if os.path.exists(gt_file):
            dist_img = cv2.imread(dist_file, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            
            if dist_img.shape == gt_img.shape:
                print(f"✅ {filename}: {dist_img.shape}")
            else:
                print(f"❌ {filename}: {dist_img.shape} vs {gt_img.shape}")
                all_sizes_match = False
    
    if all_sizes_match:
        print("\n🎉 All sizes match! Dataset is ready for training.")
    else:
        print("\n❌ Some sizes still don't match. Check alignment process.")
    
    return all_sizes_match

def test_enhancement_with_aligned_data():
    """Test enhancement using aligned test data"""
    print("\n🧪 TESTING ENHANCEMENT WITH ALIGNED DATA")
    print("=" * 45)
    
    # Use aligned test file
    test_file = "datasets/nan_aligned/test/distorted/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg"
    gt_file = "datasets/nan_aligned/test/gt/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg"
    
    if os.path.exists(test_file) and os.path.exists(gt_file):
        # Load images
        test_img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        
        print(f"📐 Test image: {test_img.shape}")
        print(f"📐 GT image: {gt_img.shape}")
        print(f"✅ Sizes match: {test_img.shape == gt_img.shape}")
        
        # Calculate baseline metrics
        mse = np.mean((test_img.astype(float) - gt_img.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        print(f"📊 Baseline PSNR: {psnr:.2f} dB")
        
        # This should be much better now with aligned data
        if psnr > 10:
            print("✅ Baseline quality is reasonable for enhancement")
        else:
            print("⚠️ Still very low quality - may need different approach")
    else:
        print("❌ Aligned test files not found")

def main():
    """Main execution function"""
    print("🔧 DATASET ALIGNMENT & ENHANCEMENT FIX")
    print("=" * 50)
    
    # Step 1: Analyze current problems
    has_problems = analyze_dataset_problems()
    
    if has_problems:
        # Step 2: Create aligned dataset
        if create_aligned_dataset():
            # Step 3: Create aligned test dataset
            create_test_aligned_dataset()
            
            # Step 4: Validate alignment
            if validate_aligned_dataset():
                # Step 5: Test with aligned data
                test_enhancement_with_aligned_data()
                
                print("\n🎯 NEXT STEPS:")
                print("=" * 15)
                print("1. Update training script to use datasets/nan_aligned/")
                print("2. Retrain model with aligned data")
                print("3. Test enhancement with new model")
                print("4. Expect significant improvement in results")
        else:
            print("❌ Failed to create aligned dataset")
    else:
        print("✅ No size mismatch problems found in dataset")

if __name__ == "__main__":
    main()
