#!/usr/bin/env python3
"""
🧪 Simple Document Enhancement Test
===================================

Test sederhana untuk enhancement document menggunakan model trained
dengan approach sliding window yang diperbaiki.

Author: Lambda One
Date: 2024
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import os

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

def simple_preprocess_document(image_path: str, segment_height: int = 128, segment_width: int = 1024, overlap: float = 0.1):
    """
    Simple sliding window preprocessing untuk document
    
    Args:
        image_path: Path ke document image
        segment_height: Height untuk setiap segment (128 untuk model)
        segment_width: Width untuk setiap segment (1024 untuk model)
        overlap: Overlap ratio antara windows
        
    Returns:
        List of segments dan koordinat mereka
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Cannot load image: {image_path}")
        return [], []
    
    print(f"📄 Image loaded: {img.shape}")
    
    # Calculate step size dengan overlap
    step_y = int(segment_height * (1 - overlap))
    step_x = int(segment_width * (1 - overlap))
    
    segments = []
    coordinates = []
    
    # Sliding window
    for y in range(0, img.shape[0] - segment_height + 1, step_y):
        for x in range(0, img.shape[1] - segment_width + 1, step_x):
            # Extract segment
            segment = img[y:y+segment_height, x:x+segment_width]
            
            # Normalize to [0, 1]
            normalized_segment = segment.astype(np.float32) / 255.0
            
            # Add channel dimension
            segment_tensor = np.expand_dims(normalized_segment, axis=-1)
            
            segments.append(segment_tensor)
            coordinates.append((y, x, y+segment_height, x+segment_width))
    
    print(f"✅ Generated {len(segments)} segments")
    return segments, coordinates

def simple_enhance_segments(generator, segments):
    """Enhance segments menggunakan generator"""
    enhanced_segments = []
    
    print(f"🎨 Enhancing {len(segments)} segments...")
    
    for i, segment in enumerate(segments):
        try:
            # Add batch dimension
            input_batch = np.expand_dims(segment, axis=0)
            
            # Enhance using generator
            enhanced_batch = generator.predict(input_batch, verbose=0)
            
            # Remove batch dimension
            enhanced_segment = enhanced_batch[0]
            
            enhanced_segments.append(enhanced_segment)
            
            if (i + 1) % 10 == 0:
                print(f"   Enhanced {i+1}/{len(segments)} segments")
                
        except Exception as e:
            print(f"⚠️ Error enhancing segment {i}: {e}")
            enhanced_segments.append(segment)
    
    print(f"✅ Enhanced {len(enhanced_segments)} segments")
    return enhanced_segments

def simple_reconstruct_document(enhanced_segments, coordinates, original_shape):
    """
    Simple reconstruction dengan blending untuk overlap areas
    """
    print(f"🔧 Reconstructing document with shape: {original_shape}")
    
    # Create output canvas
    output = np.zeros(original_shape, dtype=np.float32)
    weights = np.zeros(original_shape, dtype=np.float32)
    
    # Place enhanced segments
    for segment, (y1, x1, y2, x2) in zip(enhanced_segments, coordinates):
        # Remove channel dimension
        if len(segment.shape) == 3:
            segment_2d = segment[:, :, 0]
        else:
            segment_2d = segment
        
        # Add to output dengan weighted blending
        output[y1:y2, x1:x2] += segment_2d
        weights[y1:y2, x1:x2] += 1.0
    
    # Average overlapping areas
    valid_mask = weights > 0
    output[valid_mask] /= weights[valid_mask]
    
    # Convert to uint8
    output_uint8 = (output * 255).astype(np.uint8)
    
    return output_uint8

def test_simple_enhancement():
    """Test enhancement dengan approach yang disederhanakan"""
    print("🧪 Simple Document Enhancement Test")
    print("=" * 40)
    
    # Load generator
    weights_path = "./ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5"
    if not os.path.exists(weights_path):
        print(f"❌ Weights not found: {weights_path}")
        return
    
    print(f"🤖 Loading generator...")
    generator = build_unet_generator()
    generator.load_weights(weights_path)
    print("✅ Generator loaded!")
    
    # Test dengan different documents
    test_docs = [
        "demo_document_pipeline.png",
        "a.png",
        "b.jpg",
        "imagex.jpg"
    ]
    
    for doc_path in test_docs:
        if not os.path.exists(doc_path):
            continue
            
        print(f"\n📄 Testing: {doc_path}")
        
        try:
            # Step 1: Preprocess
            segments, coordinates = simple_preprocess_document(doc_path)
            
            if len(segments) == 0:
                print("⚠️ No segments generated")
                continue
            
            # Step 2: Enhance
            enhanced_segments = simple_enhance_segments(generator, segments)
            
            # Step 3: Reconstruct
            original_img = cv2.imread(doc_path, cv2.IMREAD_GRAYSCALE)
            reconstructed = simple_reconstruct_document(enhanced_segments, coordinates, original_img.shape)
            
            # Save results
            output_path = f"simple_enhanced_{Path(doc_path).stem}.png"
            cv2.imwrite(output_path, reconstructed)
            print(f"✅ Enhanced document saved: {output_path}")
            
            # Create comparison
            comparison_path = f"simple_comparison_{Path(doc_path).stem}.png"
            create_simple_comparison(doc_path, output_path, comparison_path)
            print(f"📊 Comparison saved: {comparison_path}")
            
        except Exception as e:
            print(f"❌ Error processing {doc_path}: {e}")

def create_simple_comparison(original_path, enhanced_path, output_path):
    """Create simple side-by-side comparison"""
    try:
        original = cv2.imread(original_path)
        enhanced = cv2.imread(enhanced_path)
        
        if original is None or enhanced is None:
            return
        
        # Resize to same height
        target_height = 600
        
        # Original
        aspect_orig = original.shape[1] / original.shape[0]
        orig_width = int(target_height * aspect_orig)
        original_resized = cv2.resize(original, (orig_width, target_height))
        
        # Enhanced
        aspect_enh = enhanced.shape[1] / enhanced.shape[0]
        enh_width = int(target_height * aspect_enh)
        enhanced_resized = cv2.resize(enhanced, (enh_width, target_height))
        
        # Comparison
        comparison = np.hstack([original_resized, enhanced_resized])
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Enhanced", (orig_width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, comparison)
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create comparison: {e}")

if __name__ == "__main__":
    test_simple_enhancement()
