#!/usr/bin/env python3
"""
Debug script untuk menganalisis mengapa hasil enhancement jadi putih
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def debug_enhancement_process():
    print("🔍 DEBUGGING ENHANCEMENT PROCESS")
    print("=" * 40)
    
    # Load images
    input_path = "datasets/nan_distorted/test/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg"
    enhanced_path = "results/enhanced_document.png"
    
    orig_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    enhanced_img = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"📂 Original image stats:")
    print(f"   Shape: {orig_img.shape}")
    print(f"   Range: {orig_img.min()} - {orig_img.max()}")
    print(f"   Mean: {orig_img.mean():.2f}")
    print(f"   Dark pixels (<128): {(orig_img < 128).sum() / orig_img.size * 100:.1f}%")
    
    print(f"\n💾 Enhanced image stats:")
    print(f"   Shape: {enhanced_img.shape}")
    print(f"   Range: {enhanced_img.min()} - {enhanced_img.max()}")
    print(f"   Mean: {enhanced_img.mean():.2f}")
    print(f"   Dark pixels (<128): {(enhanced_img < 128).sum() / enhanced_img.size * 100:.1f}%")
    
    # Coba inversi enhanced image
    inverted_enhanced = 255 - enhanced_img
    print(f"\n🔄 Inverted enhanced stats:")
    print(f"   Range: {inverted_enhanced.min()} - {inverted_enhanced.max()}")
    print(f"   Mean: {inverted_enhanced.mean():.2f}")
    print(f"   Dark pixels (<128): {(inverted_enhanced < 128).sum() / inverted_enhanced.size * 100:.1f}%")
    
    # Test model prediction raw output
    print(f"\n🤖 Testing model raw output...")
    
    # Load and preprocess
    img_resized = cv2.resize(orig_img, (128, 128))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)
    
    print(f"   Model input range: {img_input.min():.3f} - {img_input.max():.3f}")
    
    # Load model dan predict
    from document_enhancer import build_generator
    generator = build_generator()
    generator.load_weights("checkpoints/improved_model_20250814_051937/model_epoch_15_generator.weights.h5")
    
    # Raw prediction
    raw_output = generator.predict(img_input, verbose=0)
    print(f"   Model raw output range: {raw_output.min():.3f} - {raw_output.max():.3f}")
    print(f"   Model raw output mean: {raw_output.mean():.3f}")
    
    # Convert to uint8
    output_uint8 = (raw_output[0] * 255).astype(np.uint8).squeeze()
    output_resized = cv2.resize(output_uint8, (orig_img.shape[1], orig_img.shape[0]))
    
    print(f"   Final output range: {output_resized.min()} - {output_resized.max()}")
    print(f"   Final output mean: {output_resized.mean():.2f}")
    
    # Save debug visualizations
    plt.figure(figsize=(20, 4))
    
    plt.subplot(1, 5, 1)
    plt.imshow(orig_img, cmap='gray')
    plt.title(f'Original\nMean: {orig_img.mean():.1f}')
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.imshow(enhanced_img, cmap='gray')
    plt.title(f'Enhanced\nMean: {enhanced_img.mean():.1f}')
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.imshow(inverted_enhanced, cmap='gray')
    plt.title(f'Inverted Enhanced\nMean: {inverted_enhanced.mean():.1f}')
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.imshow(output_uint8, cmap='gray')
    plt.title(f'Raw Model Output\nMean: {output_uint8.mean():.1f}')
    plt.axis('off')
    
    plt.subplot(1, 5, 5)
    # Histogram comparison
    plt.hist(orig_img.flatten(), bins=50, alpha=0.7, label='Original', color='blue')
    plt.hist(enhanced_img.flatten(), bins=50, alpha=0.7, label='Enhanced', color='red')
    plt.legend()
    plt.title('Pixel Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('debug_enhancement.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Debug visualization saved: debug_enhancement.png")
    
    # Test different approaches
    print(f"\n🔧 TESTING FIXES:")
    
    # Fix 1: Invert output
    inverted_fixed = 255 - enhanced_img
    cv2.imwrite('results/enhanced_document_inverted.png', inverted_fixed)
    print(f"   ✅ Inverted version saved: results/enhanced_document_inverted.png")
    
    # Fix 2: Contrast enhancement
    contrast_enhanced = cv2.equalizeHist(enhanced_img)
    cv2.imwrite('results/enhanced_document_contrast.png', contrast_enhanced)
    print(f"   ✅ Contrast enhanced saved: results/enhanced_document_contrast.png")
    
    # Fix 3: Threshold 
    _, thresh = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('results/enhanced_document_threshold.png', thresh)
    print(f"   ✅ Threshold version saved: results/enhanced_document_threshold.png")

if __name__ == "__main__":
    debug_enhancement_process()
