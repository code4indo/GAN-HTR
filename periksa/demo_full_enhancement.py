#!/usr/bin/env python3
"""
🎯 Demo Enhanced Document Enhancement
====================================

Demo script untuk testing pipeline enhancement dokumen 
menggunakan model weights yang sudah ditraining.

Author: Lambda One
Date: 2024
"""

import os
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from document_preprocessor import DocumentPreprocessor

# Import architecture from training script
import sys
sys.path.append('.')

def build_unet_generator(input_shape=(128, 1024, 1)):
    """Build UNet Generator - EXACT architecture from training script"""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv2D, MaxPooling2D, UpSampling2D, 
        BatchNormalization, Dropout, concatenate
    )
    
    inputs = Input(shape=input_shape)
    
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
    return model

def load_generator_with_weights(weights_path: str):
    """Load generator with weights from training results"""
    try:
        print(f"🤖 Loading generator weights from: {weights_path}")
        
        # Build generator
        generator = build_unet_generator()
        
        # Load weights
        generator.load_weights(weights_path)
        
        print("✅ Generator loaded successfully!")
        return generator
        
    except Exception as e:
        print(f"❌ Error loading generator: {e}")
        return None

def enhance_image_simple(generator, image: np.ndarray) -> np.ndarray:
    """Simple enhancement using loaded generator"""
    try:
        # Preprocess image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        resized = cv2.resize(image, (1024, 128))
        
        # Normalize to [0, 1] for sigmoid output
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        input_tensor = np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)
        
        # Enhance using generator
        enhanced = generator.predict(input_tensor, verbose=0)
        
        # Denormalize from [0, 1] to [0, 255]
        enhanced = enhanced[0, :, :, 0] * 255.0
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
        
    except Exception as e:
        print(f"❌ Enhancement error: {e}")
        return image

def demo_full_pipeline():
    """Demo of the full enhancement pipeline"""
    print("🚀 Demo Full Document Enhancement Pipeline")
    print("=" * 50)
    
    # Find available model weights
    weight_dirs = [
        "./ResultGanS_S_nan_OP_SIMPLE/final/weights",
        "./ResultGanS_S_nan_OP_SIMPLE/epoch_010/weights",
        "./ResultGanS_S_nan_OP_SIMPLE/epoch_009/weights",
        "./ResultGanS_S_nan_OP_SIMPLE/epoch_008/weights",
        "./ResultGanS_S_nan_OP/final/weights"
    ]
    
    generator_weights = None
    for weight_dir in weight_dirs:
        weights_path = os.path.join(weight_dir, "generator.weights.h5")
        if os.path.exists(weights_path):
            generator_weights = weights_path
            break
    
    if not generator_weights:
        print("❌ No generator weights found!")
        print("💡 Available weight directories:")
        for weight_dir in weight_dirs:
            if os.path.exists(weight_dir):
                print(f"   • {weight_dir}")
                files = os.listdir(weight_dir)
                for f in files:
                    print(f"     - {f}")
        return
    
    print(f"🎯 Using weights: {generator_weights}")
    
    # Load generator
    generator = load_generator_with_weights(generator_weights)
    if generator is None:
        return
    
    # Create demo document if needed
    demo_doc_path = "demo_document_pipeline.png"
    if not os.path.exists(demo_doc_path):
        print("📄 Creating demo document...")
        
        # Create synthetic document
        height, width = 1200, 800
        doc = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add text lines
        lines = [
            "Enhanced Document Processing Demo",
            "Testing GAN-HTR enhancement pipeline",
            "Line-by-line enhancement approach",
            "Document reconstruction from segments",
            "Quality assessment and comparison"
        ]
        
        y_start = 150
        line_height = 120
        
        for i, line in enumerate(lines):
            y = y_start + i * line_height
            cv2.putText(doc, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, (0, 0, 0), 2)
        
        # Add noise for realism
        noise = np.random.normal(0, 15, doc.shape).astype(np.int16)
        doc = np.clip(doc.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(demo_doc_path, doc)
        print(f"✅ Demo document created: {demo_doc_path}")
    
    # Step 1: Document Preprocessing
    print("\n📋 Step 1: Document Preprocessing")
    preprocessor = DocumentPreprocessor()
    segments = preprocessor.process_document(demo_doc_path, method="sliding_window")
    
    if len(segments) == 0:
        print("⚠️ No segments generated - trying line detection...")
        segments = preprocessor.process_document(demo_doc_path, method="line_detection")
    
    print(f"✅ Generated {len(segments)} segments")
    
    # Step 2: Enhancement
    print("\n🎨 Step 2: Segment Enhancement")
    enhanced_segments = []
    
    for i, segment in enumerate(segments):
        print(f"   Enhancing segment {i+1}/{len(segments)}...")
        
        # Convert segment to proper format
        if len(segment.shape) == 3 and segment.shape[-1] == 1:
            segment_2d = segment[:, :, 0]
        else:
            segment_2d = segment
        
        # Convert from [0,1] to [0,255] if needed
        if segment_2d.max() <= 1.0:
            segment_2d = (segment_2d * 255).astype(np.uint8)
        
        # Enhance
        enhanced = enhance_image_simple(generator, segment_2d)
        enhanced_segments.append(enhanced)
    
    print(f"✅ Enhanced {len(enhanced_segments)} segments")
    
    # Step 3: Simple Reconstruction
    print("\n🔧 Step 3: Document Reconstruction")
    
    if enhanced_segments:
        # Simple vertical stacking
        line_height = 150
        max_width = max(seg.shape[1] for seg in enhanced_segments)
        
        # Resize and pad segments
        processed_segments = []
        for segment in enhanced_segments:
            # Resize to standard line height
            aspect_ratio = segment.shape[1] / segment.shape[0]
            new_width = int(line_height * aspect_ratio)
            resized = cv2.resize(segment, (new_width, line_height))
            
            # Pad to max width
            if resized.shape[1] < max_width:
                padding = max_width - resized.shape[1]
                padded = np.pad(resized, ((0, 0), (0, padding)), 
                               mode='constant', constant_values=255)
            else:
                padded = resized
            
            processed_segments.append(padded)
        
        # Stack vertically
        reconstructed = np.vstack(processed_segments)
        
        # Save results
        output_path = "enhanced_demo_document.png"
        cv2.imwrite(output_path, reconstructed)
        print(f"✅ Enhanced document saved: {output_path}")
        
        # Create comparison
        original = cv2.imread(demo_doc_path)
        enhanced_img = cv2.imread(output_path)
        
        if original is not None and enhanced_img is not None:
            # Resize for comparison
            target_height = 600
            
            # Original
            aspect_orig = original.shape[1] / original.shape[0]
            orig_width = int(target_height * aspect_orig)
            original_resized = cv2.resize(original, (orig_width, target_height))
            
            # Enhanced
            aspect_enh = enhanced_img.shape[1] / enhanced_img.shape[0]
            enh_width = int(target_height * aspect_enh)
            enhanced_resized = cv2.resize(enhanced_img, (enh_width, target_height))
            
            # Comparison
            comparison = np.hstack([original_resized, enhanced_resized])
            
            # Add labels
            cv2.putText(comparison, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "Enhanced", (orig_width + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            comparison_path = "demo_comparison.png"
            cv2.imwrite(comparison_path, comparison)
            print(f"📊 Comparison saved: {comparison_path}")
    
    print("\n🎉 Demo pipeline completed!")
    print("📁 Generated files:")
    print(f"   • Demo document: {demo_doc_path}")
    print(f"   • Enhanced document: enhanced_demo_document.png")
    print(f"   • Comparison: demo_comparison.png")

def check_available_weights():
    """Check what weights are available"""
    print("🔍 Checking available model weights...")
    
    # Search for weight files
    weight_files = []
    
    # Search in result directories
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.weights.h5') and 'generator' in file:
                full_path = os.path.join(root, file)
                weight_files.append(full_path)
    
    if weight_files:
        print(f"✅ Found {len(weight_files)} generator weight files:")
        for weight_file in weight_files:
            print(f"   • {weight_file}")
    else:
        print("❌ No generator weight files found!")
    
    return weight_files

if __name__ == "__main__":
    # Check weights first
    weights = check_available_weights()
    
    if weights:
        # Run demo
        demo_full_pipeline()
    else:
        print("\n💡 No weights found - need to train model first!")
        print("Run: python train_gan_simple.py")
