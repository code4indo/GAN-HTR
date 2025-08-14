#!/usr/bin/env python3
"""
GAN-HTR Document Enhancement Inference Script
============================================

Script untuk melakukan document enhancement/restoration menggunakan 
model Generator yang telah ditraining dengan GAN-HTR.

Features:
- Load trained Generator model
- Preprocess input documents  
- Perform enhancement/restoration
- Save enhanced results
- Batch processing support
- Quality comparison

Author: Lambda One
Date: 2024
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import argparse
import time
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# Import required modules
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, 
    Dropout, concatenate, Activation
)

def build_generator(input_size=(128, 1024, 1)):
    """
    Build UNet Generator for Document Enhancement (exact architecture from training)
    
    Args:
        input_size: Input image shape (height, width, channels)
        
    Returns:
        Generator model
    """
    inputs = Input(shape=input_size)
    
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
    
    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=output)
    return model

class DocumentEnhancer:
    """GAN-HTR Document Enhancement Engine"""
    
    def __init__(self, model_path: str, target_size: Tuple[int, int] = (1024, 128)):
        """
        Initialize Document Enhancer
        
        Args:
            model_path: Path to trained generator weights
            target_size: Target image size (width, height) - should be (1024, 128) for GAN-HTR
        """
        self.model_path = model_path
        self.target_size = target_size
        self.generator = None
        
        # Load model
        self._load_model()
        
        print(f"✅ Document Enhancer initialized!")
        print(f"   Model: {model_path}")
        print(f"   Target Size: {target_size}")
    
    def _load_model(self):
        """Load trained Generator model"""
        try:
            # Build generator architecture
            self.generator = build_generator()
            
            # Load trained weights
            self.generator.load_weights(self.model_path)
            
            print(f"✅ Generator model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess input image for enhancement
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Read image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Store original size for later
            original_size = img.shape[:2]
            
            # Resize to target size
            img_resized = cv2.resize(img, self.target_size)
            
            # Normalize to [0, 1]
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Add batch and channel dimensions
            img_tensor = np.expand_dims(img_normalized, axis=[0, -1])
            
            return img_tensor, original_size
            
        except Exception as e:
            print(f"❌ Error preprocessing image {image_path}: {e}")
            return None, None
    
    def enhance_image(self, image_tensor: np.ndarray) -> np.ndarray:
        """
        Perform document enhancement using Generator
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Enhanced image tensor
        """
        try:
            # Generate enhanced image
            enhanced_tensor = self.generator.predict(image_tensor, verbose=0)
            
            return enhanced_tensor
            
        except Exception as e:
            print(f"❌ Error during enhancement: {e}")
            return None
    
    def postprocess_image(self, enhanced_tensor: np.ndarray, 
                         original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess enhanced image tensor
        
        Args:
            enhanced_tensor: Enhanced image tensor from generator
            original_size: Original image size (height, width)
            
        Returns:
            Postprocessed image array
        """
        try:
            # Remove batch dimension
            enhanced_img = enhanced_tensor[0]
            
            # Remove channel dimension if exists
            if len(enhanced_img.shape) == 3 and enhanced_img.shape[-1] == 1:
                enhanced_img = enhanced_img[:, :, 0]
            
            # Denormalize from [0, 1] to [0, 255]
            enhanced_img = np.clip(enhanced_img * 255.0, 0, 255).astype(np.uint8)
            
            # Resize back to original size
            enhanced_img_resized = cv2.resize(enhanced_img, (original_size[1], original_size[0]))
            
            return enhanced_img_resized
            
        except Exception as e:
            print(f"❌ Error during postprocessing: {e}")
            return None
    
    def enhance_document(self, input_path: str, output_path: str, 
                        save_comparison: bool = True) -> bool:
        """
        Complete document enhancement pipeline
        
        Args:
            input_path: Path to input degraded document
            output_path: Path to save enhanced document
            save_comparison: Whether to save side-by-side comparison
            
        Returns:
            Success status
        """
        print(f"🔄 Enhancing: {input_path}")
        start_time = time.time()
        
        try:
            # Preprocess
            image_tensor, original_size = self.preprocess_image(input_path)
            if image_tensor is None:
                return False
            
            # Enhance
            enhanced_tensor = self.enhance_image(image_tensor)
            if enhanced_tensor is None:
                return False
            
            # Postprocess
            enhanced_img = self.postprocess_image(enhanced_tensor, original_size)
            if enhanced_img is None:
                return False
            
            # Save enhanced image
            cv2.imwrite(output_path, enhanced_img)
            
            # Save comparison if requested
            if save_comparison:
                self._save_comparison(input_path, output_path)
            
            processing_time = time.time() - start_time
            print(f"✅ Enhanced saved: {output_path} ({processing_time:.2f}s)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error enhancing document: {e}")
            return False
    
    def _save_comparison(self, input_path: str, output_path: str):
        """Save side-by-side comparison image"""
        try:
            # Read original and enhanced images
            original = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            enhanced = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
            
            # Create side-by-side comparison
            comparison = np.hstack([original, enhanced])
            
            # Save comparison
            comparison_path = output_path.replace('.', '_comparison.')
            cv2.imwrite(comparison_path, comparison)
            
            print(f"💾 Comparison saved: {comparison_path}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save comparison: {e}")
    
    def enhance_batch(self, input_dir: str, output_dir: str, 
                     file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
        """
        Batch enhancement of multiple documents
        
        Args:
            input_dir: Directory containing input documents
            output_dir: Directory to save enhanced documents
            file_extensions: Supported file extensions
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No image files found in {input_dir}")
            return
        
        print(f"🚀 Starting batch enhancement: {len(image_files)} files")
        
        success_count = 0
        total_time = time.time()
        
        for img_file in image_files:
            # Create output filename
            output_file = output_path / f"enhanced_{img_file.name}"
            
            # Enhance document
            if self.enhance_document(str(img_file), str(output_file)):
                success_count += 1
        
        total_time = time.time() - total_time
        
        print(f"\n🎉 Batch Enhancement Complete!")
        print(f"   ✅ Success: {success_count}/{len(image_files)} files")
        print(f"   ⏱️ Total Time: {total_time:.2f}s")
        print(f"   📁 Output: {output_dir}")

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="GAN-HTR Document Enhancement")
    
    parser.add_argument('--model', required=True, 
                       help='Path to trained generator weights (.h5)')
    parser.add_argument('--input', required=True,
                       help='Input document path or directory')
    parser.add_argument('--output', required=True,
                       help='Output path or directory')
    parser.add_argument('--batch', action='store_true',
                       help='Batch processing mode (input/output as directories)')
    parser.add_argument('--size', nargs=2, type=int, default=[1024, 128],
                       help='Target size for processing (width height) - default: 1024 128')
    parser.add_argument('--no-comparison', action='store_true',
                       help='Skip saving comparison images')
    
    args = parser.parse_args()
    
    # Initialize enhancer
    enhancer = DocumentEnhancer(
        model_path=args.model,
        target_size=tuple(args.size)
    )
    
    # Process based on mode
    if args.batch:
        # Batch processing
        enhancer.enhance_batch(args.input, args.output)
    else:
        # Single file processing
        success = enhancer.enhance_document(
            args.input, 
            args.output,
            save_comparison=not args.no_comparison
        )
        
        if success:
            print(f"\n🎉 Document enhancement completed successfully!")
        else:
            print(f"\n❌ Document enhancement failed!")
            sys.exit(1)

if __name__ == "__main__":
    print("🚀 GAN-HTR Document Enhancement Tool")
    print("=" * 50)
    
    main()
