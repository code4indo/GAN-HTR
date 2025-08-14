#!/usr/bin/env python3
"""
Test script untuk model yang sudah diperbaiki (no sigmoid saturation)
Bandingkan dengan model lama dan post-processing correction
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, Model

def create_generator_model(input_shape=(256, 256, 1)):
    """Create the same U-Net generator architecture as in training"""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder with dropout untuk prevent overfitting
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = layers.Dropout(0.1)(conv1)  # Light dropout
    pool1 = layers.MaxPooling2D(2)(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = layers.Dropout(0.1)(conv2)
    pool2 = layers.MaxPooling2D(2)(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = layers.Dropout(0.2)(conv3)
    pool3 = layers.MaxPooling2D(2)(conv3)
    
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = layers.Dropout(0.3)(conv4)
    pool4 = layers.MaxPooling2D(2)(drop4)
    
    # Bottleneck dengan heavy dropout
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = layers.Dropout(0.4)(conv5)
    
    # Decoder with skip connections
    up6 = layers.UpSampling2D(2)(drop5)
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(up6)
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    conv6 = layers.Dropout(0.3)(conv6)
    
    up7 = layers.UpSampling2D(2)(conv6)
    up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(up7)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    conv7 = layers.Dropout(0.2)(conv7)
    
    up8 = layers.UpSampling2D(2)(conv7)
    up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(up8)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    conv8 = layers.Dropout(0.1)(conv8)
    
    up9 = layers.UpSampling2D(2)(conv8)
    up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(up9)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # FIXED: Output layer dengan bias initialization untuk prevent saturation
    outputs = layers.Conv2D(1, 1, activation='sigmoid', 
                           bias_initializer='zeros',  # Prevent initial saturation
                           kernel_initializer='glorot_uniform')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Load dan preprocess gambar untuk model"""
    print(f"📁 Loading image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    print(f"📏 Original size: {image.shape}")
    
    # Resize to model input size
    image_resized = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension: (1, 256, 256, 1)
    image_batch = np.expand_dims(np.expand_dims(image_normalized, axis=0), axis=-1)
    
    print(f"🔢 Input stats:")
    print(f"  Shape: {image_batch.shape}")
    print(f"  Min: {image_batch.min():.3f}")
    print(f"  Max: {image_batch.max():.3f}")
    print(f"  Mean: {image_batch.mean():.3f}")
    
    return image_batch, image_resized

def load_fixed_model(checkpoint_path):
    """Load model yang sudah diperbaiki"""
    print(f"🤖 Loading fixed model from: {checkpoint_path}")
    
    # Create generator architecture
    generator = create_generator_model(input_shape=(256, 256, 1))
    
    # Load weights
    if os.path.exists(checkpoint_path):
        generator.load_weights(checkpoint_path)
        print("✅ Fixed model loaded successfully!")
    else:
        raise FileNotFoundError(f"Model weights not found: {checkpoint_path}")
    
    return generator

def enhance_with_fixed_model(generator, input_image):
    """Enhance image menggunakan model yang sudah diperbaiki"""
    print("🔥 Enhancing with FIXED model...")
    
    # Generate enhanced image
    enhanced = generator(input_image, training=False)
    
    # Convert to numpy
    enhanced_np = enhanced.numpy()[0, :, :, 0]  # Remove batch and channel dims
    
    print(f"📊 Fixed model output stats:")
    print(f"  Min: {enhanced_np.min():.3f}")
    print(f"  Max: {enhanced_np.max():.3f}")
    print(f"  Mean: {enhanced_np.mean():.3f}")
    
    # Check for saturation
    if enhanced_np.mean() > 0.85:
        print("⚠️  WARNING: Possible sigmoid saturation detected!")
    else:
        print("✅ No saturation detected - output in normal range")
    
    # Convert back to 0-255 range
    enhanced_uint8 = (enhanced_np * 255).astype(np.uint8)
    
    return enhanced_uint8

def load_original_model_for_comparison():
    """Load model lama untuk perbandingan (optional)"""
    try:
        print("🔄 Loading original model for comparison...")
        old_model_path = "checkpoints/generator_weights.h5"
        if os.path.exists(old_model_path):
            generator_old = create_generator_model(input_shape=(256, 256, 1))
            generator_old.load_weights(old_model_path)
            print("✅ Original model loaded for comparison")
            return generator_old
        else:
            print("ℹ️  Original model not found - skipping comparison")
            return None
    except Exception as e:
        print(f"⚠️  Could not load original model: {e}")
        return None

def save_comparison_results(input_img, fixed_output, original_output, output_dir):
    """Save hasil perbandingan"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual results
    cv2.imwrite(os.path.join(output_dir, "input_original.png"), input_img)
    cv2.imwrite(os.path.join(output_dir, "output_fixed_model.png"), fixed_output)
    
    if original_output is not None:
        cv2.imwrite(os.path.join(output_dir, "output_original_model.png"), original_output)
    
    # Create side-by-side comparison
    if original_output is not None:
        comparison = np.hstack([input_img, fixed_output, original_output])
        comparison_path = os.path.join(output_dir, "comparison_all.png")
    else:
        comparison = np.hstack([input_img, fixed_output])
        comparison_path = os.path.join(output_dir, "comparison_fixed_only.png")
    
    cv2.imwrite(comparison_path, comparison)
    print(f"💾 Comparison saved: {comparison_path}")

def main():
    parser = argparse.ArgumentParser(description="Test fixed GAN model for document enhancement")
    parser.add_argument("--input", required=True, help="Input degraded image path")
    parser.add_argument("--output", default="results/fixed_model_test", help="Output directory")
    parser.add_argument("--checkpoint", default="checkpoints/fixed_model_20250814_073224/generator_final.weights.h5", 
                       help="Fixed model checkpoint path")
    
    args = parser.parse_args()
    
    print("🚀 TESTING FIXED GAN MODEL FOR DOCUMENT ENHANCEMENT")
    print("=" * 60)
    
    try:
        # 1. Load and preprocess input
        input_batch, input_resized = load_and_preprocess_image(args.input)
        
        # 2. Load fixed model
        generator_fixed = load_fixed_model(args.checkpoint)
        
        # 3. Enhance with fixed model
        enhanced_fixed = enhance_with_fixed_model(generator_fixed, input_batch)
        
        # 4. Load original model for comparison (optional)
        generator_original = load_original_model_for_comparison()
        enhanced_original = None
        
        if generator_original is not None:
            print("🔄 Enhancing with ORIGINAL model for comparison...")
            enhanced_orig = generator_original(input_batch, training=False)
            enhanced_original = (enhanced_orig.numpy()[0, :, :, 0] * 255).astype(np.uint8)
            
            print(f"📊 Original model output stats:")
            print(f"  Min: {enhanced_orig.numpy().min():.3f}")
            print(f"  Max: {enhanced_orig.numpy().max():.3f}")
            print(f"  Mean: {enhanced_orig.numpy().mean():.3f}")
            
            if enhanced_orig.numpy().mean() > 0.85:
                print("⚠️  WARNING: Original model shows sigmoid saturation!")
        
        # 5. Save results
        save_comparison_results(
            input_resized, 
            enhanced_fixed, 
            enhanced_original, 
            args.output
        )
        
        print("\n🎯 TESTING SUMMARY:")
        print(f"📁 Input: {args.input}")
        print(f"💾 Results saved to: {args.output}/")
        print(f"🤖 Fixed model checkpoint: {args.checkpoint}")
        
        # Quality metrics
        input_mean = input_resized.mean()
        fixed_mean = enhanced_fixed.mean()
        
        print(f"\n📊 BRIGHTNESS ANALYSIS:")
        print(f"  Input mean: {input_mean:.1f}")
        print(f"  Fixed output mean: {fixed_mean:.1f}")
        print(f"  Brightness change: {((fixed_mean - input_mean) / input_mean * 100):+.1f}%")
        
        if enhanced_original is not None:
            orig_mean = enhanced_original.mean()
            print(f"  Original output mean: {orig_mean:.1f}")
            print(f"  Difference (Fixed vs Original): {fixed_mean - orig_mean:.1f}")
        
        print("\n✅ FIXED MODEL TEST COMPLETED!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
