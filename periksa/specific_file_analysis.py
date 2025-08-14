#!/usr/bin/env python3
"""
Test specific file pair mentioned by user
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

def load_and_analyze_original_sizes(distorted_path, ground_truth_path):
    """Load and analyze images at their original sizes"""
    
    print(f"🔍 ANALYSIS OF SPECIFIC FILE PAIR")
    print(f"==================================")
    print(f"📁 Distorted: {distorted_path}")
    print(f"📁 Ground truth: {ground_truth_path}")
    
    # Load images at original size
    print(f"🔄 Loading images at original size...")
    distorted_img = cv2.imread(distorted_path, cv2.IMREAD_GRAYSCALE)
    ground_truth_img = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    
    if distorted_img is None or ground_truth_img is None:
        print(f"❌ Failed to load images")
        return None
    
    print(f"📊 Image information:")
    print(f"  Distorted size: {distorted_img.shape}")
    print(f"  Ground truth size: {ground_truth_img.shape}")
    
    # Check if sizes match
    if distorted_img.shape != ground_truth_img.shape:
        print(f"⚠️  Size mismatch detected!")
        print(f"  Need to resize for comparison...")
        
        # Resize to match the smaller dimension to avoid upscaling
        target_height = min(distorted_img.shape[0], ground_truth_img.shape[0])
        target_width = min(distorted_img.shape[1], ground_truth_img.shape[1])
        target_size = (target_width, target_height)
        
        distorted_img = cv2.resize(distorted_img, target_size)
        ground_truth_img = cv2.resize(ground_truth_img, target_size)
        
        print(f"  Resized to: {distorted_img.shape}")
    
    # Calculate metrics
    print(f"📊 Calculating metrics...")
    
    baseline_psnr = calculate_psnr(distorted_img, ground_truth_img)
    baseline_ssim = calculate_ssim(distorted_img, ground_truth_img)
    mse = np.mean((distorted_img.astype(float) - ground_truth_img.astype(float)) ** 2)
    
    # Calculate additional statistics
    distorted_mean = np.mean(distorted_img)
    gt_mean = np.mean(ground_truth_img)
    distorted_std = np.std(distorted_img)
    gt_std = np.std(ground_truth_img)
    
    # Print detailed results
    print(f"\n📊 DETAILED QUALITY METRICS")
    print(f"============================")
    print(f"PSNR: {baseline_psnr:.2f} dB")
    print(f"SSIM: {baseline_ssim:.4f}")
    print(f"MSE:  {mse:.2f}")
    print(f"\nImage Statistics:")
    print(f"Distorted  - Mean: {distorted_mean:.2f}, Std: {distorted_std:.2f}")
    print(f"Ground Truth - Mean: {gt_mean:.2f}, Std: {gt_std:.2f}")
    print(f"Mean Difference: {abs(distorted_mean - gt_mean):.2f}")
    
    # Quality assessment
    if baseline_psnr < 10:
        quality_assessment = "Very Poor"
        enhancement_need = "Critical"
    elif baseline_psnr < 20:
        quality_assessment = "Poor"
        enhancement_need = "High"
    elif baseline_psnr < 30:
        quality_assessment = "Fair"
        enhancement_need = "Medium"
    elif baseline_psnr < 40:
        quality_assessment = "Good"
        enhancement_need = "Low"
    else:
        quality_assessment = "Excellent"
        enhancement_need = "None"
    
    print(f"\nQuality Assessment: {quality_assessment}")
    print(f"Enhancement Need: {enhancement_need}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Detailed Analysis: User Specified File Pair', fontsize=16, fontweight='bold')
    
    # Row 1: Original images
    axes[0, 0].imshow(distorted_img, cmap='gray')
    axes[0, 0].set_title(f'Distorted Image\nSize: {distorted_img.shape}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ground_truth_img, cmap='gray')
    axes[0, 1].set_title(f'Ground Truth Image\nSize: {ground_truth_img.shape}')
    axes[0, 1].axis('off')
    
    # Row 2: Analysis maps
    diff_map = np.abs(distorted_img.astype(float) - ground_truth_img.astype(float))
    
    im1 = axes[1, 0].imshow(diff_map, cmap='hot')
    axes[1, 0].set_title(f'Absolute Difference Map\nMax Diff: {np.max(diff_map):.1f}')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Normalized cross-correlation
    norm_distorted = (distorted_img - distorted_mean) / distorted_std
    norm_gt = (ground_truth_img - gt_mean) / gt_std
    correlation = norm_distorted * norm_gt
    
    im2 = axes[1, 1].imshow(correlation, cmap='RdBu', vmin=-3, vmax=3)
    axes[1, 1].set_title(f'Normalized Correlation Map\nMean: {np.mean(correlation):.3f}')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Row 3: Statistical analysis
    axes[2, 0].hist(distorted_img.flatten(), bins=50, alpha=0.7, label='Distorted', color='red', density=True)
    axes[2, 0].hist(ground_truth_img.flatten(), bins=50, alpha=0.7, label='Ground Truth', color='blue', density=True)
    axes[2, 0].set_title('Pixel Intensity Distribution')
    axes[2, 0].set_xlabel('Pixel Intensity')
    axes[2, 0].set_ylabel('Density')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Error distribution
    error_map = distorted_img.astype(float) - ground_truth_img.astype(float)
    axes[2, 1].hist(error_map.flatten(), bins=50, color='orange', alpha=0.7, density=True)
    axes[2, 1].set_title('Error Distribution')
    axes[2, 1].set_xlabel('Error (Distorted - Ground Truth)')
    axes[2, 1].set_ylabel('Density')
    axes[2, 1].axvline(0, color='black', linestyle='--', alpha=0.7)
    axes[2, 1].grid(True, alpha=0.3)
    
    # Add comprehensive metrics text
    metrics_text = (f'PSNR: {baseline_psnr:.2f} dB\n'
                   f'SSIM: {baseline_ssim:.4f}\n'
                   f'MSE: {mse:.2f}\n'
                   f'Quality: {quality_assessment}\n'
                   f'Enhancement Need: {enhancement_need}\n\n'
                   f'Mean Intensity:\n'
                   f'  Distorted: {distorted_mean:.1f}\n'
                   f'  Ground Truth: {gt_mean:.1f}\n'
                   f'  Difference: {abs(distorted_mean - gt_mean):.1f}')
    
    fig.text(0.02, 0.02, metrics_text, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save results
    os.makedirs("specific_file_analysis", exist_ok=True)
    output_file = "specific_file_analysis/user_specified_file_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Results saved to: {output_file}")
    
    plt.show()
    
    # Generate enhancement potential analysis
    print(f"\n🎯 ENHANCEMENT POTENTIAL ANALYSIS")
    print(f"=================================")
    
    potential_gain = 0
    if baseline_psnr < 15:
        potential_gain = 10-15  # Significant improvement possible
        print(f"💡 Significant enhancement potential: {potential_gain} dB PSNR improvement possible")
    elif baseline_psnr < 25:
        potential_gain = 5-10
        print(f"💡 Moderate enhancement potential: {potential_gain} dB PSNR improvement possible")
    else:
        potential_gain = 2-5
        print(f"💡 Limited enhancement potential: {potential_gain} dB PSNR improvement possible")
    
    print(f"📈 Target PSNR range: {baseline_psnr + potential_gain:.1f} dB")
    print(f"📈 Target SSIM range: {min(0.95, baseline_ssim + 0.3):.3f}")
    
    return {
        'baseline_psnr': baseline_psnr,
        'baseline_ssim': baseline_ssim,
        'baseline_mse': mse,
        'quality_assessment': quality_assessment,
        'enhancement_need': enhancement_need,
        'potential_psnr_gain': potential_gain,
        'distorted_stats': {'mean': distorted_mean, 'std': distorted_std},
        'gt_stats': {'mean': gt_mean, 'std': gt_std},
        'image_size': distorted_img.shape
    }

def main():
    # Test the specific file pair mentioned by user
    distorted_file = "datasets/nan_distorted/test/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg"
    ground_truth_file = "datasets/nan_raw_biner/test/images/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg"
    
    print("🎯 ANALYSIS OF USER SPECIFIED FILE PAIR")
    print("=" * 50)
    print(f"Target file: 001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg")
    
    if os.path.exists(distorted_file) and os.path.exists(ground_truth_file):
        results = load_and_analyze_original_sizes(distorted_file, ground_truth_file)
        
        if results:
            print(f"\n✅ Analysis completed successfully!")
            print(f"🔑 Key findings:")
            print(f"   - Current quality: {results['quality_assessment']}")
            print(f"   - Enhancement priority: {results['enhancement_need']}")
            print(f"   - Improvement potential: {results['potential_psnr_gain']} dB")
            
            # Diagnosis of why enhancement might not be working well
            print(f"\n🔍 ENHANCEMENT MODEL DIAGNOSIS")
            print(f"==============================")
            if results['baseline_psnr'] < 10:
                print(f"❌ Extremely low baseline PSNR ({results['baseline_psnr']:.2f} dB)")
                print(f"   → Model needs to improve by {20 - results['baseline_psnr']:.1f} dB to reach 'fair' quality")
                print(f"   → This requires learning very complex degradation patterns")
                
            if results['baseline_ssim'] < 0.3:
                print(f"❌ Very low structural similarity ({results['baseline_ssim']:.4f})")
                print(f"   → Images have fundamentally different structures")
                print(f"   → Model needs to reconstruct missing information, not just enhance")
                
            print(f"\n💡 RECOMMENDATIONS FOR BETTER ENHANCEMENT")
            print(f"=========================================")
            print(f"1. Train with more diverse degradation patterns")
            print(f"2. Use larger training dataset with similar degradation types")
            print(f"3. Implement progressive enhancement (multi-stage)")
            print(f"4. Consider domain adaptation techniques")
            print(f"5. Use perceptual loss in addition to pixel-wise loss")
    
    else:
        print(f"❌ Files not found:")
        print(f"   Distorted: {os.path.exists(distorted_file)}")
        print(f"   Ground truth: {os.path.exists(ground_truth_file)}")

if __name__ == "__main__":
    main()
