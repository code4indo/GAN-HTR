#!/usr/bin/env python3
"""
🚀 GAN-HTR Quick Start Guide - Document Enhancement
=================================================

This script provides the fastest way to get started with document enhancement
using your trained GAN-HTR model.

Features:
- One-command document enhancement
- Automatic model detection
- Built-in examples
- Error handling and guidance

Author: Lambda One
Date: 2024
"""

import os
import sys

def print_header():
    """Print welcome header"""
    print("🚀 GAN-HTR Document Enhancement - Quick Start")
    print("=" * 55)
    print()

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking Requirements...")
    
    # Check if model exists
    model_path = "ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5"
    if not os.path.exists(model_path):
        print("❌ Model not found!")
        print(f"   Expected: {model_path}")
        print("   Please ensure training was completed successfully.")
        return False
    
    # Check if enhancement script exists
    if not os.path.exists("enhance_document.py"):
        print("❌ Enhancement script not found!")
        print("   Please ensure enhance_document.py is in current directory.")
        return False
    
    # Check for sample images
    sample_images = ["a.png", "b.jpg"]
    found_samples = [img for img in sample_images if os.path.exists(img)]
    
    if not found_samples:
        print("⚠️ No sample images found for testing.")
        print(f"   Looking for: {', '.join(sample_images)}")
    else:
        print(f"✅ Found sample images: {', '.join(found_samples)}")
    
    print("✅ All requirements met!")
    print()
    return True

def show_quick_examples():
    """Show quick usage examples"""
    print("⚡ Quick Examples:")
    print("-" * 30)
    
    print("\n1. 📄 Single Document Enhancement:")
    print("   poetry run python enhance_document.py \\")
    print("     --model ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5 \\")
    print("     --input your_document.jpg \\")
    print("     --output enhanced_document.png")
    
    print("\n2. 📁 Batch Processing:")
    print("   poetry run python enhance_document.py \\")
    print("     --model ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5 \\")
    print("     --input input_folder/ \\")
    print("     --output output_folder/ \\")
    print("     --batch")
    
    print("\n3. 🎛️ Custom Settings:")
    print("   poetry run python enhance_document.py \\")
    print("     --model ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5 \\")
    print("     --input document.jpg \\")
    print("     --output enhanced.png \\")
    print("     --size 1024 128 \\")
    print("     --no-comparison")
    print()

def run_sample_test():
    """Run a sample test if possible"""
    print("🧪 Sample Test:")
    print("-" * 20)
    
    # Find first available sample
    sample_images = ["a.png", "b.jpg", "imagex.jpg"]
    sample_image = None
    
    for img in sample_images:
        if os.path.exists(img):
            sample_image = img
            break
    
    if not sample_image:
        print("⚠️ No sample images available for testing.")
        print("   Place a document image (a.png, b.jpg, or imagex.jpg) to test.")
        return False
    
    print(f"🔄 Testing with: {sample_image}")
    output_image = f"quick_test_enhanced_{sample_image.split('.')[0]}.png"
    
    # Build command
    cmd = f"poetry run python enhance_document.py --model ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5 --input {sample_image} --output {output_image}"
    
    print(f"   Running: {cmd}")
    print()
    
    # Run the command
    result = os.system(cmd)
    
    if result == 0:
        print("✅ Sample enhancement successful!")
        print(f"   Output: {output_image}")
        print(f"   Comparison: {output_image.replace('.png', '_comparison.png')}")
        return True
    else:
        print("❌ Sample enhancement failed!")
        return False

def show_model_info():
    """Show information about available models"""
    print("📊 Available Models:")
    print("-" * 25)
    
    models_info = [
        ("🥇 FINAL (Recommended)", "ResultGanS_S_nan_OP_SIMPLE/final/weights/generator.weights.h5", "Best overall performance"),
        ("🥈 Epoch 9", "ResultGanS_S_nan_OP_SIMPLE/epoch_009/weights/generator.weights.h5", "Conservative enhancement"),
        ("🥉 Epoch 8", "ResultGanS_S_nan_OP_SIMPLE/epoch_008/weights/generator.weights.h5", "Balanced performance"),
    ]
    
    for name, path, desc in models_info:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"   {exists} {name}")
        print(f"      Path: {path}")
        print(f"      Description: {desc}")
        print()

def show_tips():
    """Show usage tips"""
    print("💡 Pro Tips:")
    print("-" * 15)
    print("• Use PNG format for better quality output")
    print("• Model works best with 1024x128 size (default)")
    print("• Comparison images help visualize improvements")
    print("• Use --batch for processing multiple documents")
    print("• Check GPU memory usage with nvidia-smi")
    print("• For best results, use clear scan images")
    print()

def show_troubleshooting():
    """Show common troubleshooting tips"""
    print("🔧 Troubleshooting:")
    print("-" * 20)
    print("• Shape mismatch error: Ensure --size 1024 128")
    print("• Out of memory: Reduce batch size or image resolution")
    print("• Model not found: Check path and training completion")
    print("• TensorFlow warnings: Usually safe to ignore")
    print("• Slow performance: Ensure GPU drivers are updated")
    print()

def main():
    """Main function"""
    print_header()
    
    # Check requirements
    if not check_requirements():
        print("❌ Please fix the issues above before continuing.")
        return
    
    # Show model information
    show_model_info()
    
    # Show examples
    show_quick_examples()
    
    # Show tips
    show_tips()
    
    # Ask if user wants to run sample test
    print("🤔 Would you like to run a sample test? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice in ['y', 'yes']:
            print()
            if run_sample_test():
                print("🎉 Quick start completed successfully!")
            else:
                print("⚠️ Sample test failed, but you can still use the enhancement script manually.")
        else:
            print("👍 Skipping sample test. You can run enhancement manually using the examples above.")
    except KeyboardInterrupt:
        print("\n👋 Exiting quick start guide.")
        return
    
    print()
    show_troubleshooting()
    
    print("📚 For more information:")
    print("   • Read: MODEL_ANALYSIS_DOCUMENT_ENHANCEMENT.md")
    print("   • Read: TRAINING_SUCCESS_SUMMARY.md")
    print("   • Run: python enhance_document.py --help")
    print()
    print("🎯 Happy document enhancing! 🚀")

if __name__ == "__main__":
    main()
