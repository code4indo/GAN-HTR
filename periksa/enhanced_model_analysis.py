#!/usr/bin/env python3
"""
🔍 Enhanced Model Analysis Tool
===============================

Tool untuk menganalisis kualitas model enhancement dengan lebih detail
dan membandingkan dengan ground truth.

Author: Lambda One
Date: August 14, 2025
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
import argparse

def calculate_metrics(original, enhanced, ground_truth):
    """Calculate comprehensive quality metrics"""
    metrics = {}
    
    # PSNR
    def psnr(img1, img2):
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    # SSIM
    def calculate_ssim(img1, img2):
        return ssim(img1, img2, data_range=255)
    
    # Mean Square Error
    def mse(img1, img2):
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    # Baseline (Original vs Ground Truth)
    metrics['baseline_psnr'] = psnr(original, ground_truth)
    metrics['baseline_ssim'] = calculate_ssim(original, ground_truth)
    metrics['baseline_mse'] = mse(original, ground_truth)
    
    # Enhanced vs Ground Truth
    metrics['enhanced_psnr'] = psnr(enhanced, ground_truth)
    metrics['enhanced_ssim'] = calculate_ssim(enhanced, ground_truth)
    metrics['enhanced_mse'] = mse(enhanced, ground_truth)
    
    # Original vs Enhanced
    metrics['improvement_psnr'] = metrics['enhanced_psnr'] - metrics['baseline_psnr']
    metrics['improvement_ssim'] = metrics['enhanced_ssim'] - metrics['baseline_ssim']
    metrics['improvement_mse'] = metrics['baseline_mse'] - metrics['enhanced_mse']  # Lower is better
    
    return metrics

def load_generator_model(model_path):
    """Load generator model with proper architecture"""
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Conv2D, MaxPooling2D, UpSampling2D, 
            BatchNormalization, Dropout, concatenate
        )
        
        inputs = Input(shape=(128, 1024, 1))
        
        # Encoder
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # Bottleneck
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.3)(conv4)

        # Decoder
        up5 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop4))
        up5 = BatchNormalization()(up5)
        merge5 = concatenate([conv3, up5])
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)

        up6 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
        up6 = BatchNormalization()(up6)
        merge6 = concatenate([conv2, up6])
        conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        up7 = BatchNormalization()(up7)
        merge7 = concatenate([conv1, up7])
        conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
        
        output = Conv2D(1, 1, activation='sigmoid')(conv7)

        model = Model(inputs=inputs, outputs=output)
        
        # Load weights
        model.load_weights(model_path)
        print(f"✅ Model loaded successfully: {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def enhance_image(generator, image):
    """Enhance image using generator"""
    try:
        # Preprocess
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        resized = cv2.resize(image, (1024, 128))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        input_tensor = np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)
        
        # Enhance
        enhanced = generator.predict(input_tensor, verbose=0)
        
        # Denormalize
        enhanced = enhanced[0, :, :, 0] * 255.0
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Resize back to original size
        enhanced_resized = cv2.resize(enhanced, (image.shape[1], image.shape[0]))
        
        return enhanced_resized
        
    except Exception as e:
        print(f"❌ Enhancement error: {e}")
        return image

def analyze_specific_file(distorted_path, gt_path, model_path, output_dir="analysis_results"):
    """Analyze specific file pair"""
    print(f"🔍 Analyzing: {os.path.basename(distorted_path)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    if not os.path.exists(distorted_path):
        print(f"❌ Distorted image not found: {distorted_path}")
        return None
        
    if not os.path.exists(gt_path):
        print(f"❌ Ground truth image not found: {gt_path}")
        return None
    
    distorted = cv2.imread(distorted_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    if distorted is None or gt is None:
        print("❌ Failed to load images")
        return None
    
    print(f"📐 Image sizes - Distorted: {distorted.shape}, GT: {gt.shape}")
    
    # Resize if needed
    if distorted.shape != gt.shape:
        print("⚠️ Size mismatch, resizing distorted to match GT")
        distorted = cv2.resize(distorted, (gt.shape[1], gt.shape[0]))
    
    # Load and test model
    generator = load_generator_model(model_path)
    if generator is None:
        return None
    
    # Enhance image
    print("🎨 Enhancing image...")
    enhanced = enhance_image(generator, distorted)
    
    # Calculate metrics
    print("📊 Calculating metrics...")
    metrics = calculate_metrics(distorted, enhanced, gt)
    
    # Print results
    print("\n📈 QUALITY METRICS:")
    print("="*50)
    print(f"📋 Baseline (Original vs GT):")
    print(f"   PSNR: {metrics['baseline_psnr']:.2f} dB")
    print(f"   SSIM: {metrics['baseline_ssim']:.4f}")
    print(f"   MSE:  {metrics['baseline_mse']:.2f}")
    print()
    print(f"🎯 Enhanced (Enhanced vs GT):")
    print(f"   PSNR: {metrics['enhanced_psnr']:.2f} dB")
    print(f"   SSIM: {metrics['enhanced_ssim']:.4f}")
    print(f"   MSE:  {metrics['enhanced_mse']:.2f}")
    print()
    print(f"📈 Improvement:")
    print(f"   PSNR: {metrics['improvement_psnr']:+.2f} dB")
    print(f"   SSIM: {metrics['improvement_ssim']:+.4f}")
    print(f"   MSE:  {metrics['improvement_mse']:+.2f} (lower is better)")
    
    # Quality assessment
    if metrics['improvement_psnr'] > 1.0:
        print("✅ SIGNIFICANT IMPROVEMENT in PSNR")
    elif metrics['improvement_psnr'] > 0:
        print("🟡 MINOR IMPROVEMENT in PSNR")
    else:
        print("❌ NO IMPROVEMENT or DEGRADATION in PSNR")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Enhancement Analysis: {os.path.basename(distorted_path)}', fontsize=16)
    
    # Row 1: Images
    axes[0, 0].imshow(distorted, cmap='gray')
    axes[0, 0].set_title(f'Original\nPSNR: {metrics["baseline_psnr"]:.2f} dB')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(enhanced, cmap='gray')
    axes[0, 1].set_title(f'Enhanced\nPSNR: {metrics["enhanced_psnr"]:.2f} dB')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gt, cmap='gray')
    axes[0, 2].set_title('Ground Truth')
    axes[0, 2].axis('off')
    
    # Row 2: Differences
    diff_orig = np.abs(distorted.astype(float) - gt.astype(float))
    diff_enh = np.abs(enhanced.astype(float) - gt.astype(float))
    diff_improvement = diff_orig - diff_enh
    
    axes[1, 0].imshow(diff_orig, cmap='hot')
    axes[1, 0].set_title(f'Original Error\nMSE: {metrics["baseline_mse"]:.2f}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff_enh, cmap='hot')
    axes[1, 1].set_title(f'Enhanced Error\nMSE: {metrics["enhanced_mse"]:.2f}')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(diff_improvement, cmap='RdYlBu_r')
    axes[1, 2].set_title(f'Improvement Map\nΔMSE: {metrics["improvement_mse"]:.2f}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save results
    filename = os.path.basename(distorted_path).replace('.jpg', '').replace('.png', '')
    
    # Save visualization
    viz_path = os.path.join(output_dir, f"analysis_{filename}.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual images
    cv2.imwrite(os.path.join(output_dir, f"original_{filename}.png"), distorted)
    cv2.imwrite(os.path.join(output_dir, f"enhanced_{filename}.png"), enhanced)
    cv2.imwrite(os.path.join(output_dir, f"ground_truth_{filename}.png"), gt)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"metrics_{filename}.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Enhancement Analysis: {filename}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Baseline (Original vs GT):\n")
        f.write(f"  PSNR: {metrics['baseline_psnr']:.2f} dB\n")
        f.write(f"  SSIM: {metrics['baseline_ssim']:.4f}\n")
        f.write(f"  MSE:  {metrics['baseline_mse']:.2f}\n\n")
        f.write(f"Enhanced (Enhanced vs GT):\n")
        f.write(f"  PSNR: {metrics['enhanced_psnr']:.2f} dB\n")
        f.write(f"  SSIM: {metrics['enhanced_ssim']:.4f}\n")
        f.write(f"  MSE:  {metrics['enhanced_mse']:.2f}\n\n")
        f.write(f"Improvement:\n")
        f.write(f"  PSNR: {metrics['improvement_psnr']:+.2f} dB\n")
        f.write(f"  SSIM: {metrics['improvement_ssim']:+.4f}\n")
        f.write(f"  MSE:  {metrics['improvement_mse']:+.2f}\n")
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   📊 Visualization: {viz_path}")
    print(f"   📋 Metrics: {metrics_path}")
    
    return metrics

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description="Enhanced Model Analysis Tool")
    parser.add_argument('--distorted', required=True, help='Path to distorted image')
    parser.add_argument('--ground_truth', required=True, help='Path to ground truth image')
    parser.add_argument('--model', required=True, help='Path to generator model weights')
    parser.add_argument('--output', default='analysis_results', help='Output directory')
    
    args = parser.parse_args()
    
    print("🔍 ENHANCED MODEL ANALYSIS TOOL")
    print("=" * 50)
    
    metrics = analyze_specific_file(
        args.distorted, 
        args.ground_truth, 
        args.model, 
        args.output
    )
    
    if metrics:
        print("\n🎯 ANALYSIS COMPLETE!")
    else:
        print("\n❌ ANALYSIS FAILED!")

if __name__ == "__main__":
    main()
