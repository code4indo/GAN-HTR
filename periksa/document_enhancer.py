#!/usr/bin/env python3
"""
Document Enhancement Tool menggunakan GAN-HTR Model
Restorasi dokumen dengan model yang sudah di-training

Usage:
    python document_enhancer.py --input path/to/degraded_image.jpg --output path/to/enhanced_image.jpg
    python document_enhancer.py --input path/to/image.jpg  # Auto save ke enhanced_[filename]
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

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess image untuk model input"""
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    original_size = img.shape
    print(f"📐 Original size: {original_size}")
    
    # Resize untuk model (128x128)
    img_resized = cv2.resize(img, target_size)
    
    # Normalize ke [0,1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dan channel dimensions
    img_input = np.expand_dims(img_normalized, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)
    
    return img_input, img, original_size

def postprocess_image(enhanced_output, original_size):
    """Postprocess model output ke image format"""
    # Remove batch dimension dan convert ke uint8
    enhanced = (enhanced_output[0] * 255).astype(np.uint8).squeeze()
    
    # Resize kembali ke ukuran asli
    if original_size != (128, 128):
        enhanced = cv2.resize(enhanced, (original_size[1], original_size[0]))
    
    return enhanced

def calculate_metrics(img1, img2):
    """Calculate PSNR dan SSIM"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # PSNR
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # SSIM
    ssim_val = ssim(img1, img2, data_range=255)
    
    return psnr, ssim_val

def enhance_document(input_path, output_path=None, model_path=None, show_comparison=True):
    """
    Main function untuk document enhancement
    
    Args:
        input_path: Path ke gambar yang akan di-enhance
        output_path: Path output (optional, auto-generate jika None)
        model_path: Path ke model weights (optional, gunakan default)
        show_comparison: Show comparison plot
    """
    
    print("🔍 GAN-HTR DOCUMENT ENHANCER")
    print("=" * 35)
    
    # Default paths
    if model_path is None:
        model_path = "checkpoints/improved_model_20250814_051937/model_epoch_15_generator.weights.h5"
    
    if output_path is None:
        input_dir = os.path.dirname(input_path)
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(input_dir, f"enhanced_{input_name}.png")
    
    # Check files
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    print(f"📂 Input: {input_path}")
    print(f"💾 Output: {output_path}")
    print(f"🤖 Model: {model_path}")
    
    # Load dan preprocess image
    print("\n📂 Loading and preprocessing image...")
    img_input, original_img, original_size = preprocess_image(input_path)
    
    # Build model
    print("🏗️ Building model...")
    generator = build_generator()
    
    # Load weights
    print("📥 Loading trained weights...")
    try:
        generator.load_weights(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    
    # Generate enhanced image
    print("🚀 Enhancing document...")
    enhanced_output = generator.predict(img_input, verbose=0)
    enhanced_img = postprocess_image(enhanced_output, original_size)
    
    # Save result
    cv2.imwrite(output_path, enhanced_img)
    print(f"💾 Enhanced image saved: {output_path}")
    
    # Calculate improvement metrics (if possible)
    print(f"\n📊 ENHANCEMENT COMPLETED")
    print(f"Original size: {original_size}")
    print(f"Enhanced image saved successfully!")
    
    # Show comparison jika diminta
    if show_comparison:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title('Original (Degraded)')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(enhanced_img, cmap='gray')
        plt.title('Enhanced')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        diff = np.abs(enhanced_img.astype(float) - original_img.astype(float))
        plt.imshow(diff, cmap='hot')
        plt.title('Difference Map')
        plt.axis('off')
        plt.colorbar()
        
        plt.tight_layout()
        
        # Save comparison
        comparison_path = output_path.replace('.png', '_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"📊 Comparison saved: {comparison_path}")
        
        # Show plot
        plt.show()
    
    return enhanced_img, original_img

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='GAN-HTR Document Enhancement Tool')
    parser.add_argument('--input', '-i', required=True, help='Path to input degraded image')
    parser.add_argument('--output', '-o', help='Path to output enhanced image (optional)')
    parser.add_argument('--model', '-m', help='Path to model weights (optional)')
    parser.add_argument('--no-comparison', action='store_true', help='Skip comparison visualization')
    
    args = parser.parse_args()
    
    try:
        enhanced_img, original_img = enhance_document(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            show_comparison=not args.no_comparison
        )
        print("\n✅ Document enhancement completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
