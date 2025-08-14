#!/usr/bin/env python3
"""
Example Usage of GAN-HTR Document Enhancement
==========================================

Demo script untuk menjalankan document enhancement dengan berbagai cara.

Usage Examples:
1. Single document enhancement
2. Batch processing
3. Quality comparison
4. Model performance testing

Author: Lambda One
Date: 2024
"""

import os
import sys
from pathlib import Path

def test_single_enhancement():
    """Test enhancement on single document"""
    print("📄 Testing Single Document Enhancement")
    print("=" * 50)
    
    # Parameters
    model_path = "ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5"
    input_image = "a.png"  # Example degraded document
    output_image = "enhanced_a.png"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please check the model path from training results")
        return False
    
    if not os.path.exists(input_image):
        print(f"❌ Input image not found: {input_image}")
        print("Please provide a test image")
        return False
    
    # Run enhancement
    cmd = f"python enhance_document.py --model {model_path} --input {input_image} --output {output_image}"
    print(f"🚀 Running: {cmd}")
    
    result = os.system(cmd)
    
    if result == 0:
        print(f"✅ Enhancement successful! Output: {output_image}")
        return True
    else:
        print(f"❌ Enhancement failed!")
        return False

def test_batch_enhancement():
    """Test batch enhancement on multiple documents"""
    print("\n📁 Testing Batch Document Enhancement")
    print("=" * 50)
    
    # Parameters
    model_path = "ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5"
    input_dir = "datasets/anriRusak/"
    output_dir = "enhanced_documents/"
    
    # Check if directories exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run batch enhancement
    cmd = f"python enhance_document.py --model {model_path} --input {input_dir} --output {output_dir} --batch"
    print(f"🚀 Running batch processing: {cmd}")
    
    result = os.system(cmd)
    
    if result == 0:
        print(f"✅ Batch enhancement successful! Output directory: {output_dir}")
        return True
    else:
        print(f"❌ Batch enhancement failed!")
        return False

def test_different_models():
    """Test different epoch models for comparison"""
    print("\n🔄 Testing Different Epoch Models")
    print("=" * 50)
    
    input_image = "a.png"
    
    if not os.path.exists(input_image):
        print(f"❌ Input image not found: {input_image}")
        return False
    
    # Test different epochs
    epochs_to_test = [8, 9, 10, "final"]
    
    for epoch in epochs_to_test:
        if epoch == "final":
            model_path = "ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5"
            output_image = f"enhanced_a_final.png"
        else:
            model_path = f"ResultGanS_S_nan_OP_SIMPLE/epoch_{epoch:03d}/weights/generator.weights.h5"
            output_image = f"enhanced_a_epoch_{epoch}.png"
        
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found: {model_path}")
            continue
        
        print(f"🔄 Testing Epoch {epoch}...")
        cmd = f"python enhance_document.py --model {model_path} --input {input_image} --output {output_image} --no-comparison"
        
        result = os.system(cmd)
        
        if result == 0:
            print(f"✅ Epoch {epoch} enhancement successful!")
        else:
            print(f"❌ Epoch {epoch} enhancement failed!")

def create_comparison_grid():
    """Create comparison grid of different models"""
    print("\n🖼️ Creating Comparison Grid")
    print("=" * 50)
    
    try:
        import cv2
        import numpy as np
        
        # Input files (from previous test)
        original = "a.png"
        enhanced_files = [
            "enhanced_a_epoch_8.png",
            "enhanced_a_epoch_9.png", 
            "enhanced_a_epoch_10.png",
            "enhanced_a_final.png"
        ]
        
        # Check if original exists
        if not os.path.exists(original):
            print(f"❌ Original image not found: {original}")
            return False
        
        # Load original
        orig_img = cv2.imread(original, cv2.IMREAD_GRAYSCALE)
        if orig_img is None:
            print(f"❌ Cannot read original image: {original}")
            return False
        
        # Load enhanced images
        enhanced_imgs = []
        labels = ["Original", "Epoch 8", "Epoch 9", "Epoch 10", "Final"]
        
        enhanced_imgs.append(orig_img)
        
        for enhanced_file in enhanced_files:
            if os.path.exists(enhanced_file):
                img = cv2.imread(enhanced_file, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    enhanced_imgs.append(img)
                else:
                    print(f"⚠️ Cannot read: {enhanced_file}")
            else:
                print(f"⚠️ File not found: {enhanced_file}")
        
        if len(enhanced_imgs) < 2:
            print("❌ Need at least original + 1 enhanced image")
            return False
        
        # Resize all images to same size
        target_height = 200
        resized_imgs = []
        
        for img in enhanced_imgs:
            aspect_ratio = img.shape[1] / img.shape[0]
            target_width = int(target_height * aspect_ratio)
            resized = cv2.resize(img, (target_width, target_height))
            resized_imgs.append(resized)
        
        # Create grid
        max_width = max(img.shape[1] for img in resized_imgs)
        
        # Pad images to same width
        padded_imgs = []
        for i, img in enumerate(resized_imgs):
            if img.shape[1] < max_width:
                padding = max_width - img.shape[1]
                img = np.pad(img, ((0, 0), (0, padding)), mode='constant', constant_values=255)
            
            # Add label
            label = labels[i] if i < len(labels) else f"Enhanced {i}"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
            padded_imgs.append(img)
        
        # Stack vertically
        comparison_grid = np.vstack(padded_imgs)
        
        # Save comparison
        output_file = "model_comparison_grid.png"
        cv2.imwrite(output_file, comparison_grid)
        
        print(f"✅ Comparison grid saved: {output_file}")
        return True
        
    except ImportError:
        print("⚠️ OpenCV not available for grid creation")
        return False
    except Exception as e:
        print(f"❌ Error creating comparison grid: {e}")
        return False

def main():
    """Main demo function"""
    print("🚀 GAN-HTR Document Enhancement Demo")
    print("=" * 60)
    
    # Check if enhancement script exists
    if not os.path.exists("enhance_document.py"):
        print("❌ enhance_document.py not found!")
        print("Please ensure the enhancement script is in the current directory")
        return
    
    # Run tests
    tests = [
        ("Single Document Enhancement", test_single_enhancement),
        ("Different Model Comparison", test_different_models),
        ("Comparison Grid Creation", create_comparison_grid),
        ("Batch Processing", test_batch_enhancement)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 DEMO RESULTS SUMMARY")
    print("="*60)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    passed_tests = sum(1 for success in results.values() if success)
    total_tests = len(results)
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All demos completed successfully!")
        print("\n📋 NEXT STEPS:")
        print("1. Use enhanced_document.py for your document enhancement needs")
        print("2. Experiment with different epoch models for best results")
        print("3. Adjust preprocessing parameters for your specific documents")
        print("4. Consider fine-tuning for your specific document types")
    else:
        print("⚠️ Some demos failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
