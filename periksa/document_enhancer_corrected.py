#!/usr/bin/env python3
"""
Corrected Document Enhancer - Mengatasi masalah output terlalu putih
"""

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def build_generator():
    """Build U-Net generator - sama dengan yang digunakan saat training"""
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model
    
    inputs = layers.Input(shape=(128, 128, 1))
    
    # Encoder (downsampling)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(2)(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(2)(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(2)(conv3)
    
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(2)(drop4)
    
    # Bottleneck
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)
    
    # Decoder (upsampling)
    up6 = layers.UpSampling2D(2)(drop5)
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(up6)
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = layers.UpSampling2D(2)(conv6)
    up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(up7)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = layers.UpSampling2D(2)(conv7)
    up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(up8)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = layers.UpSampling2D(2)(conv8)
    up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(up9)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output layer with sigmoid
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def correct_brightness(enhanced_img, target_mean=199.6):
    """
    Corrected brightness untuk match dengan expected GT mean
    """
    current_mean = enhanced_img.mean()
    
    if current_mean > 250:  # Jika terlalu putih
        # Metode 1: Linear scaling ke target range
        corrected = enhanced_img * (target_mean / 255.0)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        # Jika masih kurang bagus, coba contrast adjustment
        if corrected.mean() < target_mean * 0.8:
            # Metode 2: Histogram mapping
            corrected = cv2.equalizeHist(corrected)
            
            # Scale to target mean
            scale_factor = target_mean / corrected.mean()
            corrected = np.clip(corrected * scale_factor, 0, 255).astype(np.uint8)
    else:
        corrected = enhanced_img
    
    return corrected

def enhance_document_corrected(input_path, output_path=None, model_path=None, show_comparison=True):
    """
    Enhanced document enhancement dengan brightness correction
    """
    
    print("🔍 GAN-HTR CORRECTED DOCUMENT ENHANCER")
    print("=" * 40)
    
    # Default paths
    if model_path is None:
        model_path = "checkpoints/improved_model_20250814_051937/model_epoch_15_generator.weights.h5"
    
    if output_path is None:
        input_dir = os.path.dirname(input_path)
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(input_dir, f"corrected_{input_name}.png")
    
    print(f"📂 Input: {input_path}")
    print(f"💾 Output: {output_path}")
    
    # Load image
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    original_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    original_size = original_img.shape
    print(f"📐 Original size: {original_size}")
    print(f"📊 Original stats: mean={original_img.mean():.1f}, range={original_img.min()}-{original_img.max()}")
    
    # Preprocess untuk model
    img_resized = cv2.resize(original_img, (128, 128))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)
    
    # Load model
    print("🏗️ Building and loading model...")
    generator = build_generator()
    generator.load_weights(model_path)
    
    # Generate enhancement
    print("🚀 Generating enhancement...")
    raw_output = generator.predict(img_input, verbose=0)
    
    # Post-process dengan correction
    enhanced_128 = (raw_output[0] * 255).astype(np.uint8).squeeze()
    enhanced_resized = cv2.resize(enhanced_128, (original_size[1], original_size[0]))
    
    print(f"📊 Raw model output: mean={enhanced_resized.mean():.1f}")
    
    # Apply brightness correction
    print("🔧 Applying brightness correction...")
    corrected_img = correct_brightness(enhanced_resized, target_mean=199.6)
    
    print(f"📊 Corrected output: mean={corrected_img.mean():.1f}")
    
    # Save result
    cv2.imwrite(output_path, corrected_img)
    print(f"💾 Corrected enhanced image saved: {output_path}")
    
    # Comparison dengan ground truth jika ada
    gt_path = input_path.replace('nan_distorted/test/', 'nan_raw_biner/test/images/')
    if os.path.exists(gt_path):
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt_img, (original_size[1], original_size[0]))
        
        # Calculate metrics
        from skimage.metrics import structural_similarity as ssim
        
        def calculate_psnr(img1, img2):
            mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            if mse == 0:
                return float('inf')
            return 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Original vs GT
        orig_psnr = calculate_psnr(original_img, gt_resized)
        orig_ssim = ssim(original_img, gt_resized, data_range=255)
        
        # Corrected vs GT
        corr_psnr = calculate_psnr(corrected_img, gt_resized)
        corr_ssim = ssim(corrected_img, gt_resized, data_range=255)
        
        print(f"\n📊 METRICS COMPARISON")
        print(f"Original vs GT:  PSNR={orig_psnr:.2f} dB, SSIM={orig_ssim:.4f}")
        print(f"Corrected vs GT: PSNR={corr_psnr:.2f} dB, SSIM={corr_ssim:.4f}")
        print(f"Improvement:     PSNR={corr_psnr-orig_psnr:+.2f} dB, SSIM={corr_ssim-orig_ssim:+.4f}")
    
    if show_comparison:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title(f'Original\\nMean: {original_img.mean():.1f}')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(enhanced_resized, cmap='gray')
        plt.title(f'Raw Enhanced\\nMean: {enhanced_resized.mean():.1f}')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(corrected_img, cmap='gray')
        plt.title(f'Corrected\\nMean: {corrected_img.mean():.1f}')
        plt.axis('off')
        
        if os.path.exists(gt_path):
            plt.subplot(1, 4, 4)
            plt.imshow(gt_resized, cmap='gray')
            plt.title(f'Ground Truth\\nMean: {gt_resized.mean():.1f}')
            plt.axis('off')
        
        plt.tight_layout()
        comparison_path = output_path.replace('.png', '_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"📊 Comparison saved: {comparison_path}")
        plt.show()
    
    return corrected_img

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Corrected GAN-HTR Document Enhancement Tool')
    parser.add_argument('--input', '-i', required=True, help='Path to input degraded image')
    parser.add_argument('--output', '-o', help='Path to output enhanced image (optional)')
    parser.add_argument('--model', '-m', help='Path to model weights (optional)')
    parser.add_argument('--no-comparison', action='store_true', help='Skip comparison visualization')
    
    args = parser.parse_args()
    
    try:
        enhanced_img = enhance_document_corrected(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            show_comparison=not args.no_comparison
        )
        print("\\n✅ Corrected document enhancement completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
