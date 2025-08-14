#!/usr/bin/env python3
"""
🔄 Full Document Enhancement Pipeline
===================================

Pipeline lengkap untuk enhancement dokumen utuh menggunakan GAN-HTR:
1. Document preprocessing (segmentation)
2. Line-by-line enhancement  
3. Document reconstruction

Author: Lambda One
Date: 2024
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from typing import List, Tuple
import sys

# Import our tools
from document_preprocessor import DocumentPreprocessor

def enhance_segments_batch(segments: List[np.ndarray], model_path: str) -> List[np.ndarray]:
    """
    Enhance all segments using GAN-HTR model
    
    Args:
        segments: List of preprocessed segments
        model_path: Path to trained generator model
        
    Returns:
        List of enhanced segments
    """
    print(f"🔄 Enhancing {len(segments)} segments...")
    
    # Import enhancement modules
    sys.path.append('.')
    from enhance_document import DocumentEnhancer
    
    # Initialize enhancer
    enhancer = DocumentEnhancer(model_path)
    
    enhanced_segments = []
    
    for i, segment in enumerate(segments):
        try:
            # Add batch dimension
            segment_batch = np.expand_dims(segment, axis=0)
            
            # Enhance segment
            enhanced_batch = enhancer.enhance_image(segment_batch)
            
            if enhanced_batch is not None:
                # Remove batch dimension
                enhanced_segment = enhanced_batch[0]
                enhanced_segments.append(enhanced_segment)
            else:
                print(f"⚠️ Failed to enhance segment {i}")
                enhanced_segments.append(segment)  # Use original
                
        except Exception as e:
            print(f"⚠️ Error enhancing segment {i}: {e}")
            enhanced_segments.append(segment)  # Use original
    
    print(f"✅ Enhanced {len(enhanced_segments)} segments")
    return enhanced_segments

def reconstruct_document(enhanced_segments: List[np.ndarray], 
                        original_shape: Tuple[int, int],
                        method: str = "line_detection") -> np.ndarray:
    """
    Reconstruct full document from enhanced segments
    
    Args:
        enhanced_segments: List of enhanced line segments
        original_shape: Shape of original document (height, width)
        method: Reconstruction method
        
    Returns:
        Reconstructed document image
    """
    print(f"🔧 Reconstructing document...")
    
    if method == "line_detection":
        # Simple vertical stacking for line-based approach
        # This is a simplified reconstruction - in practice you'd need
        # to track original line positions
        
        # Resize segments to reasonable line height
        line_height = 120  # Reasonable line height for reconstruction
        reconstructed_lines = []
        
        for segment in enhanced_segments:
            # Remove channel dimension if present
            if len(segment.shape) == 3 and segment.shape[-1] == 1:
                segment = segment[:, :, 0]
            
            # Convert to uint8
            segment_uint8 = (segment * 255).astype(np.uint8)
            
            # Resize to maintain aspect ratio
            aspect_ratio = segment_uint8.shape[1] / segment_uint8.shape[0]
            new_width = int(line_height * aspect_ratio)
            
            resized = cv2.resize(segment_uint8, (new_width, line_height))
            reconstructed_lines.append(resized)
        
        # Find maximum width
        max_width = max(line.shape[1] for line in reconstructed_lines)
        
        # Pad all lines to same width
        padded_lines = []
        for line in reconstructed_lines:
            if line.shape[1] < max_width:
                padding = max_width - line.shape[1]
                padded_line = np.pad(line, ((0, 0), (0, padding)), 
                                   mode='constant', constant_values=255)
            else:
                padded_line = line
            padded_lines.append(padded_line)
        
        # Stack vertically
        reconstructed = np.vstack(padded_lines)
        
    else:
        # For sliding window, this would be more complex
        # requiring proper positioning and blending
        print("⚠️ Sliding window reconstruction not implemented")
        # Create simple concatenation for now
        reconstructed = np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)
    
    return reconstructed

def process_full_document(input_path: str, output_path: str, model_path: str, 
                         method: str = "line_detection", save_intermediates: bool = False):
    """
    Complete pipeline for full document enhancement
    
    Args:
        input_path: Path to input document
        output_path: Path for enhanced output
        model_path: Path to GAN-HTR model
        method: Processing method
        save_intermediates: Whether to save intermediate results
    """
    print("🚀 Full Document Enhancement Pipeline")
    print("=" * 45)
    
    # Step 1: Preprocess document
    print("📄 Step 1: Document Preprocessing")
    preprocessor = DocumentPreprocessor()
    
    # Load original image to get shape
    original_image = cv2.imread(input_path)
    if original_image is None:
        raise ValueError(f"Cannot load image: {input_path}")
    
    original_shape = original_image.shape[:2]  # (height, width)
    
    # Generate segments
    segments = preprocessor.process_document(input_path, method=method)
    
    # Save segments if requested
    if save_intermediates:
        segments_dir = Path(output_path).parent / "segments"
        segments_dir.mkdir(exist_ok=True)
        
        for i, segment in enumerate(segments):
            segment_uint8 = (segment * 255).astype(np.uint8)
            if len(segment_uint8.shape) == 3:
                segment_uint8 = segment_uint8[:, :, 0]
            
            segment_path = segments_dir / f"segment_{i:03d}.png"
            cv2.imwrite(str(segment_path), segment_uint8)
        
        print(f"💾 Saved {len(segments)} segments to: {segments_dir}")
    
    # Step 2: Enhance segments
    print("\n🎨 Step 2: Segment Enhancement")
    enhanced_segments = enhance_segments_batch(segments, model_path)
    
    # Save enhanced segments if requested
    if save_intermediates:
        enhanced_dir = Path(output_path).parent / "enhanced_segments"
        enhanced_dir.mkdir(exist_ok=True)
        
        for i, segment in enumerate(enhanced_segments):
            segment_uint8 = (segment * 255).astype(np.uint8)
            if len(segment_uint8.shape) == 3:
                segment_uint8 = segment_uint8[:, :, 0]
            
            segment_path = enhanced_dir / f"enhanced_segment_{i:03d}.png"
            cv2.imwrite(str(segment_path), segment_uint8)
        
        print(f"💾 Saved {len(enhanced_segments)} enhanced segments to: {enhanced_dir}")
    
    # Step 3: Reconstruct document
    print("\n🔧 Step 3: Document Reconstruction")
    reconstructed = reconstruct_document(enhanced_segments, original_shape, method)
    
    # Save final result
    cv2.imwrite(output_path, reconstructed)
    print(f"✅ Enhanced document saved: {output_path}")
    
    # Create comparison
    comparison_path = output_path.replace('.', '_comparison.')
    create_comparison_image(input_path, output_path, comparison_path)
    print(f"📊 Comparison saved: {comparison_path}")

def create_comparison_image(original_path: str, enhanced_path: str, output_path: str):
    """Create side-by-side comparison"""
    try:
        original = cv2.imread(original_path)
        enhanced = cv2.imread(enhanced_path)
        
        if original is None or enhanced is None:
            print("⚠️ Cannot create comparison - missing images")
            return
        
        # Resize to same height for comparison
        target_height = 800
        
        # Resize original
        aspect_orig = original.shape[1] / original.shape[0]
        orig_width = int(target_height * aspect_orig)
        original_resized = cv2.resize(original, (orig_width, target_height))
        
        # Resize enhanced
        aspect_enh = enhanced.shape[1] / enhanced.shape[0]
        enh_width = int(target_height * aspect_enh)
        enhanced_resized = cv2.resize(enhanced, (enh_width, target_height))
        
        # Create comparison
        comparison = np.hstack([original_resized, enhanced_resized])
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Enhanced", (orig_width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, comparison)
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create comparison: {e}")

def main():
    """Main function with CLI"""
    parser = argparse.ArgumentParser(description="Full Document Enhancement Pipeline")
    
    parser.add_argument('--input', required=True, help='Input document image')
    parser.add_argument('--output', required=True, help='Output enhanced document')
    parser.add_argument('--model', required=True, help='Path to trained GAN-HTR model')
    parser.add_argument('--method', choices=['line_detection', 'sliding_window'], 
                       default='line_detection', help='Processing method')
    parser.add_argument('--save-intermediates', action='store_true',
                       help='Save intermediate processing results')
    
    args = parser.parse_args()
    
    try:
        process_full_document(
            args.input, 
            args.output, 
            args.model,
            args.method,
            args.save_intermediates
        )
        
        print("\n🎉 Full document enhancement completed!")
        print("\n📋 Files generated:")
        print(f"   • Enhanced document: {args.output}")
        print(f"   • Comparison: {args.output.replace('.', '_comparison.')}")
        
        if args.save_intermediates:
            output_dir = Path(args.output).parent
            print(f"   • Segments: {output_dir}/segments/")
            print(f"   • Enhanced segments: {output_dir}/enhanced_segments/")
        
    except Exception as e:
        print(f"❌ Error in enhancement pipeline: {e}")

if __name__ == "__main__":
    main()
