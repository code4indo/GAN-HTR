#!/usr/bin/env python3
"""
🔧 Document Preprocessing untuk GAN-HTR Enhancement
==================================================

Script untuk memproses dokumen utuh menjadi segmen yang kompatibel 
dengan model GAN-HTR yang ditraining pada line-level.

Problem:
- Training: Word/line segments (~1800x110 pixels, landscape)
- Inference: Full documents (~3100x4700 pixels, portrait)

Solution:
- Line segmentation + individual enhancement
- Sliding window approach
- Patch-based processing

Author: Lambda One
Date: 2024
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from typing import List, Tuple
import matplotlib.pyplot as plt

class DocumentPreprocessor:
    """Preprocess full documents for GAN-HTR line-level enhancement"""
    
    def __init__(self, target_line_height: int = 128, target_line_width: int = 1024):
        """
        Initialize preprocessor
        
        Args:
            target_line_height: Expected line height for model
            target_line_width: Expected line width for model
        """
        self.target_line_height = target_line_height
        self.target_line_width = target_line_width
        
    def detect_text_lines(self, image: np.ndarray, min_line_height: int = 30) -> List[Tuple[int, int, int, int]]:
        """
        Detect text lines in document using projection methods
        
        Args:
            image: Input document image
            min_line_height: Minimum height for valid text line
            
        Returns:
            List of (x, y, w, h) bounding boxes for detected lines
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection to find text lines
        horizontal_projection = np.sum(binary, axis=1)
        
        # Find line boundaries
        lines = []
        in_line = False
        line_start = 0
        
        for i, projection in enumerate(horizontal_projection):
            if projection > 0 and not in_line:
                # Start of new line
                line_start = i
                in_line = True
            elif projection == 0 and in_line:
                # End of line
                line_height = i - line_start
                if line_height >= min_line_height:
                    lines.append((0, line_start, image.shape[1], line_height))
                in_line = False
        
        # Handle case where last line goes to bottom
        if in_line:
            line_height = len(horizontal_projection) - line_start
            if line_height >= min_line_height:
                lines.append((0, line_start, image.shape[1], line_height))
        
        return lines
    
    def extract_line_segments(self, image: np.ndarray, padding: int = 10) -> List[np.ndarray]:
        """
        Extract individual line segments from document
        
        Args:
            image: Input document image
            padding: Padding around detected lines
            
        Returns:
            List of line segment images
        """
        lines = self.detect_text_lines(image)
        segments = []
        
        for x, y, w, h in lines:
            # Add padding
            y_start = max(0, y - padding)
            y_end = min(image.shape[0], y + h + padding)
            x_start = max(0, x)
            x_end = min(image.shape[1], x + w)
            
            # Extract segment
            segment = image[y_start:y_end, x_start:x_end]
            segments.append(segment)
        
        return segments
    
    def preprocess_line_for_model(self, line_image: np.ndarray) -> np.ndarray:
        """
        Preprocess single line to match model expectations
        
        Args:
            line_image: Input line image
            
        Returns:
            Preprocessed line image ready for model
        """
        # Convert to grayscale if needed
        if len(line_image.shape) == 3:
            gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_image.copy()
        
        # Resize to target dimensions
        resized = cv2.resize(gray, (self.target_line_width, self.target_line_height))
        
        # Normalize
        normalized = resized.astype('float32') / 255.0
        
        # Add channel dimension
        model_input = np.expand_dims(normalized, axis=-1)
        
        return model_input
    
    def sliding_window_segments(self, image: np.ndarray, 
                              window_height: int = 128, 
                              window_width: int = 1024,
                              overlap: float = 0.2) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Create sliding window segments for large documents
        
        Args:
            image: Input document image
            window_height: Height of sliding window
            window_width: Width of sliding window  
            overlap: Overlap ratio between windows
            
        Returns:
            List of (segment_image, (start_y, start_x)) tuples
        """
        segments = []
        
        # Calculate step sizes
        step_y = int(window_height * (1 - overlap))
        step_x = int(window_width * (1 - overlap))
        
        # Generate sliding windows
        for y in range(0, image.shape[0] - window_height + 1, step_y):
            for x in range(0, image.shape[1] - window_width + 1, step_x):
                # Extract window
                window = image[y:y+window_height, x:x+window_width]
                
                # Convert to grayscale if needed
                if len(window.shape) == 3:
                    window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                
                segments.append((window, (y, x)))
        
        return segments
    
    def process_document(self, image_path: str, method: str = "line_detection") -> List[np.ndarray]:
        """
        Process full document into segments suitable for GAN-HTR
        
        Args:
            image_path: Path to input document
            method: Processing method ("line_detection" or "sliding_window")
            
        Returns:
            List of preprocessed segments ready for model
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        print(f"📄 Processing: {image_path}")
        print(f"   Original size: {image.shape[1]} x {image.shape[0]}")
        
        if method == "line_detection":
            # Extract text lines
            line_segments = self.extract_line_segments(image)
            print(f"   Detected {len(line_segments)} text lines")
            
            # Preprocess each line
            processed_segments = []
            for i, segment in enumerate(line_segments):
                processed = self.preprocess_line_for_model(segment)
                processed_segments.append(processed)
            
        elif method == "sliding_window":
            # Use sliding window approach
            window_segments = self.sliding_window_segments(image)
            print(f"   Generated {len(window_segments)} sliding windows")
            
            # Preprocess each window
            processed_segments = []
            for segment, position in window_segments:
                # Normalize and add channel dimension
                normalized = segment.astype('float32') / 255.0
                model_input = np.expand_dims(normalized, axis=-1)
                processed_segments.append(model_input)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return processed_segments
    
    def visualize_segmentation(self, image_path: str, output_path: str):
        """
        Visualize line detection results
        
        Args:
            image_path: Path to input document
            output_path: Path to save visualization
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Detect lines
        lines = self.detect_text_lines(image)
        
        # Draw detected lines
        vis_image = image.copy()
        for i, (x, y, w, h) in enumerate(lines):
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_image, f"Line {i+1}", (x + 10, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save visualization
        cv2.imwrite(output_path, vis_image)
        print(f"📊 Segmentation visualization saved: {output_path}")

def main():
    """Main function with CLI"""
    parser = argparse.ArgumentParser(description="Document Preprocessing for GAN-HTR")
    
    parser.add_argument('--input', required=True, help='Input document image path')
    parser.add_argument('--output', help='Output directory for segments')
    parser.add_argument('--method', choices=['line_detection', 'sliding_window'], 
                       default='line_detection', help='Segmentation method')
    parser.add_argument('--visualize', action='store_true', 
                       help='Save segmentation visualization')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DocumentPreprocessor()
    
    try:
        # Process document
        segments = preprocessor.process_document(args.input, method=args.method)
        
        print(f"✅ Generated {len(segments)} segments ready for GAN-HTR")
        print(f"   Segment size: {segments[0].shape} each")
        
        # Save segments if output directory specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, segment in enumerate(segments):
                # Convert back to uint8 for saving
                segment_uint8 = (segment * 255).astype(np.uint8)
                if len(segment_uint8.shape) == 3:
                    segment_uint8 = segment_uint8[:, :, 0]
                
                segment_path = output_dir / f"segment_{i:03d}.png"
                cv2.imwrite(str(segment_path), segment_uint8)
            
            print(f"💾 Segments saved to: {args.output}")
        
        # Create visualization if requested
        if args.visualize:
            vis_path = args.input.replace('.jpg', '_segmentation.jpg')
            preprocessor.visualize_segmentation(args.input, vis_path)
        
        print("\n📋 NEXT STEPS:")
        print("1. Use generated segments with enhance_document.py")
        print("2. Process each segment individually") 
        print("3. Combine enhanced segments back to full document")
        
    except Exception as e:
        print(f"❌ Error processing document: {e}")

if __name__ == "__main__":
    print("🔧 Document Preprocessor for GAN-HTR")
    print("=" * 45)
    main()
