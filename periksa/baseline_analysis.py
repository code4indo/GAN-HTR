#!/usr/bin/env python3
"""
Simple test using existing GAN-HTR model weights
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Add project root to path
sys.path.append('/home/lambda_one/tesis/GAN-HTR')

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    return ssim(img1, img2, data_range=255)

def load_image_for_analysis(image_path, target_size=(128, 128)):
    """Load image for analysis"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Resize to target size
    original_size = img.shape
    img = cv2.resize(img, target_size)
    
    return img, original_size

def analyze_images_without_model(distorted_path, ground_truth_path, output_dir):
    """Analyze image pair without model enhancement"""
    
    print(f"🔍 Baseline Image Analysis")
    print(f"==========================")
    print(f"📁 Distorted: {distorted_path}")
    print(f"📁 Ground truth: {ground_truth_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    print(f"🔄 Loading images...")
    distorted_result = load_image_for_analysis(distorted_path)
    ground_truth_result = load_image_for_analysis(ground_truth_path)
    
    if distorted_result is None or ground_truth_result is None:
        print(f"❌ Failed to load images")
        return None
    
    distorted_img, distorted_orig_size = distorted_result
    ground_truth_img, gt_orig_size = ground_truth_result
    
    print(f"📊 Image information:")
    print(f"  Distorted original size: {distorted_orig_size}")
    print(f"  Ground truth original size: {gt_orig_size}")
    print(f"  Processed size: {distorted_img.shape}")
    
    # Calculate metrics
    print(f"📊 Calculating metrics...")
    
    # Baseline: Distorted vs Ground Truth
    baseline_psnr = calculate_psnr(distorted_img, ground_truth_img)
    baseline_ssim = calculate_ssim(distorted_img, ground_truth_img)
    
    # Calculate MSE
    mse = np.mean((distorted_img.astype(float) - ground_truth_img.astype(float)) ** 2)
    
    # Print results
    print(f"\n📊 BASELINE QUALITY METRICS")
    print(f"============================")
    print(f"PSNR: {baseline_psnr:.2f} dB")
    print(f"SSIM: {baseline_ssim:.4f}")
    print(f"MSE:  {mse:.2f}")
    
    # Analyze the quality
    if baseline_psnr < 10:
        quality_assessment = "Very Poor"
    elif baseline_psnr < 20:
        quality_assessment = "Poor"
    elif baseline_psnr < 30:
        quality_assessment = "Fair"
    elif baseline_psnr < 40:
        quality_assessment = "Good"
    else:
        quality_assessment = "Excellent"
    
    print(f"Quality Assessment: {quality_assessment}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Baseline Image Analysis (No Enhancement)', fontsize=16, fontweight='bold')
    
    # Top row: Images
    axes[0, 0].imshow(distorted_img, cmap='gray')
    axes[0, 0].set_title(f'Distorted Input\nOriginal Size: {distorted_orig_size}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ground_truth_img, cmap='gray')
    axes[0, 1].set_title(f'Ground Truth\nOriginal Size: {gt_orig_size}')
    axes[0, 1].axis('off')
    
    # Bottom row: Analysis
    diff_map = np.abs(distorted_img.astype(float) - ground_truth_img.astype(float))
    
    im1 = axes[1, 0].imshow(diff_map, cmap='hot')
    axes[1, 0].set_title('Difference Map')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Histogram
    axes[1, 1].hist(distorted_img.flatten(), bins=50, alpha=0.7, label='Distorted', color='red')
    axes[1, 1].hist(ground_truth_img.flatten(), bins=50, alpha=0.7, label='Ground Truth', color='blue')
    axes[1, 1].set_title('Pixel Intensity Distribution')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    # Add metrics text
    metrics_text = f'PSNR: {baseline_psnr:.2f} dB\nSSIM: {baseline_ssim:.4f}\nMSE: {mse:.2f}\nQuality: {quality_assessment}'
    fig.text(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    # Save results
    output_file = os.path.join(output_dir, f'baseline_analysis_{Path(distorted_path).stem}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Results saved to: {output_file}")
    
    plt.show()
    
    return {
        'baseline_psnr': baseline_psnr,
        'baseline_ssim': baseline_ssim,
        'baseline_mse': mse,
        'quality_assessment': quality_assessment,
        'distorted_orig_size': distorted_orig_size,
        'gt_orig_size': gt_orig_size
    }

def test_multiple_pairs(test_dir, output_dir, max_pairs=5):
    """Test multiple image pairs"""
    
    distorted_dir = os.path.join(test_dir, "distorted")
    gt_dir = os.path.join(test_dir, "gt")
    
    if not os.path.exists(distorted_dir) or not os.path.exists(gt_dir):
        print(f"❌ Test directories not found")
        return
    
    distorted_files = list(Path(distorted_dir).glob("*.jpg"))[:max_pairs]
    
    results = []
    
    print(f"🚀 TESTING MULTIPLE IMAGE PAIRS")
    print(f"================================")
    print(f"Testing {len(distorted_files)} pairs...")
    
    for i, distorted_file in enumerate(distorted_files, 1):
        gt_file = os.path.join(gt_dir, distorted_file.name)
        
        if os.path.exists(gt_file):
            print(f"\n--- Pair {i}/{len(distorted_files)}: {distorted_file.name} ---")
            result = analyze_images_without_model(
                str(distorted_file),
                gt_file,
                os.path.join(output_dir, f"pair_{i}")
            )
            if result:
                result['filename'] = distorted_file.name
                results.append(result)
    
    # Summary statistics
    if results:
        print(f"\n📊 SUMMARY STATISTICS")
        print(f"======================")
        
        psnr_values = [r['baseline_psnr'] for r in results]
        ssim_values = [r['baseline_ssim'] for r in results]
        mse_values = [r['baseline_mse'] for r in results]
        
        print(f"PSNR - Mean: {np.mean(psnr_values):.2f} dB, Std: {np.std(psnr_values):.2f} dB")
        print(f"SSIM - Mean: {np.mean(ssim_values):.4f}, Std: {np.std(ssim_values):.4f}")
        print(f"MSE  - Mean: {np.mean(mse_values):.2f}, Std: {np.std(mse_values):.2f}")
        
        # Find best and worst cases
        best_psnr_idx = np.argmax(psnr_values)
        worst_psnr_idx = np.argmin(psnr_values)
        
        print(f"\nBest case: {results[best_psnr_idx]['filename']} (PSNR: {psnr_values[best_psnr_idx]:.2f} dB)")
        print(f"Worst case: {results[worst_psnr_idx]['filename']} (PSNR: {psnr_values[worst_psnr_idx]:.2f} dB)")
    
    return results

def main():
    print("🚀 BASELINE ANALYSIS OF NaN DATASET")
    print("=" * 40)
    
    # Test multiple pairs
    results = test_multiple_pairs(
        "datasets/nan_aligned/test",
        "baseline_analysis_results",
        max_pairs=5
    )
    
    if results:
        print(f"\n✅ Analysis completed for {len(results)} image pairs")
        
        # Look for the specific file mentioned by user
        target_file = "001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg"
        for result in results:
            if target_file in result['filename']:
                print(f"🎯 Found target file: {result['filename']}")
                print(f"   PSNR: {result['baseline_psnr']:.2f} dB")
                print(f"   SSIM: {result['baseline_ssim']:.4f}")
                break
    else:
        print("❌ No valid image pairs found for analysis")

if __name__ == "__main__":
    main()
