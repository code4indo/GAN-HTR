#!/usr/bin/env python3
"""
Simple test dengan trained model - Compatible version
"""

import os
import sys
import tensorflow as tf
import numpy as np
import cv2
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

def build_compatible_generator():
    """Build generator yang sama persis dengan train_improved_model.py - U-Net"""
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
    from tensorflow.keras.models import Model
    from tensorflow.keras import layers
    
    inputs = layers.Input(shape=(128, 128, 1))
    
    # Encoder (downsampling) - sama seperti di train_improved_model.py
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
    
    # Decoder (upsampling) - sama seperti di train_improved_model.py
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
    
    # Output layer with sigmoid for better text enhancement
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def test_enhancement():
    """Test enhancement dengan model yang trained"""
    
    print("🔍 TESTING ENHANCED MODEL")
    print("=" * 30)
    
    # File paths
    distorted_file = "datasets/nan_distorted/test/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg"
    gt_file = "datasets/nan_raw_biner/test/images/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg"
    model_weights = "checkpoints/improved_model_20250814_051937/model_epoch_15_generator.weights.h5"
    
    # Check files exist
    if not all(os.path.exists(f) for f in [distorted_file, gt_file, model_weights]):
        print("❌ Some files not found!")
        return
    
    # Load images
    print("📂 Loading images...")
    distorted_img = cv2.imread(distorted_file, cv2.IMREAD_GRAYSCALE)
    gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
    
    print(f"📐 Original sizes - Distorted: {distorted_img.shape}, GT: {gt_img.shape}")
    
    # Resize to same size for comparison
    target_size = (min(distorted_img.shape[1], gt_img.shape[1]), 
                   min(distorted_img.shape[0], gt_img.shape[0]))
    
    distorted_resized = cv2.resize(distorted_img, target_size)
    gt_resized = cv2.resize(gt_img, target_size)
    
    print(f"📐 Resized to: {distorted_resized.shape}")
    
    # Prepare for model input (128x128)
    distorted_model_input = cv2.resize(distorted_resized, (128, 128))
    distorted_model_input = distorted_model_input.astype(np.float32) / 255.0
    distorted_model_input = np.expand_dims(distorted_model_input, axis=-1)
    distorted_model_input = np.expand_dims(distorted_model_input, axis=0)
    
    # Build model
    print("🏗️ Building model...")
    generator = build_compatible_generator()
    
    # Load weights
    print("📥 Loading trained weights...")
    try:
        generator.load_weights(model_weights)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Generate enhanced image
    print("🚀 Generating enhanced image...")
    enhanced_output = generator.predict(distorted_model_input, verbose=0)
    enhanced_128 = (enhanced_output[0] * 255).astype(np.uint8).squeeze()
    
    # Resize enhanced back to comparison size
    enhanced_resized = cv2.resize(enhanced_128, target_size)
    
    # Calculate metrics
    print("📊 Calculating metrics...")
    
    # Baseline: Distorted vs GT
    baseline_psnr = calculate_psnr(distorted_resized, gt_resized)
    baseline_ssim = calculate_ssim(distorted_resized, gt_resized)
    
    # Enhanced: Enhanced vs GT
    enhanced_psnr = calculate_psnr(enhanced_resized, gt_resized)
    enhanced_ssim = calculate_ssim(enhanced_resized, gt_resized)
    
    # Results
    print(f"\n📊 ENHANCEMENT RESULTS")
    print(f"======================")
    print(f"Baseline (Distorted vs GT):")
    print(f"  PSNR: {baseline_psnr:.2f} dB")
    print(f"  SSIM: {baseline_ssim:.4f}")
    print(f"\nEnhanced (Enhanced vs GT):")
    print(f"  PSNR: {enhanced_psnr:.2f} dB")
    print(f"  SSIM: {enhanced_ssim:.4f}")
    print(f"\nImprovement:")
    print(f"  PSNR: {enhanced_psnr - baseline_psnr:+.2f} dB")
    print(f"  SSIM: {enhanced_ssim - baseline_ssim:+.4f}")
    
    # Save visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('GAN-HTR Enhancement Results', fontsize=16)
    
    axes[0].imshow(distorted_resized, cmap='gray')
    axes[0].set_title(f'Distorted\nPSNR: {baseline_psnr:.2f} dB')
    axes[0].axis('off')
    
    axes[1].imshow(enhanced_resized, cmap='gray')
    axes[1].set_title(f'Enhanced\nPSNR: {enhanced_psnr:.2f} dB')
    axes[1].axis('off')
    
    axes[2].imshow(gt_resized, cmap='gray')
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    # Difference map
    diff = np.abs(enhanced_resized.astype(float) - gt_resized.astype(float))
    im = axes[3].imshow(diff, cmap='hot')
    axes[3].set_title('Enhancement Error Map')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3])
    
    # Save results
    os.makedirs("test_results", exist_ok=True)
    plt.savefig("test_results/enhancement_test.png", dpi=200, bbox_inches='tight')
    cv2.imwrite("test_results/enhanced_output.png", enhanced_resized)
    
    print(f"\n💾 Results saved to test_results/")
    print(f"✅ TESTING COMPLETED!")

if __name__ == "__main__":
    test_enhancement()
