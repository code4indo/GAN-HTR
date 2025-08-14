#!/usr/bin/env python3
"""
🧪 Test Full Document Enhancement Pipeline
=========================================

Script untuk menguji pipeline enhancement dokumen utuh
dengan berbagai sample images dan settings.

Author: Lambda One
Date: 2024
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil

def test_with_sample_documents():
    """Test the full document enhancement pipeline"""
    print("🧪 Testing Full Document Enhancement Pipeline")
    print("=" * 50)
    
    # Find available models
    model_files = list(Path('.').glob('generator_model_epoch_*.h5'))
    if not model_files:
        print("❌ No trained models found!")
        return
    
    # Use the latest model
    latest_model = sorted(model_files)[-1]
    print(f"🤖 Using model: {latest_model}")
    
    # Find test documents
    test_documents = []
    
    # Look for documents in various locations
    search_paths = [
        'datasets/anriRusak',
        'datasets/nan_raw_color',
        '.'
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                files = list(Path(search_path).glob(ext))
                test_documents.extend(files[:3])  # Take first 3 from each location
    
    # Also check root directory for any sample images
    root_images = [f for f in Path('.').glob('*.jpg')] + [f for f in Path('.').glob('*.png')]
    test_documents.extend(root_images[:2])
    
    if not test_documents:
        print("❌ No test documents found!")
        print("💡 Place some document images in the current directory to test")
        return
    
    # Remove duplicates and take first 3
    test_documents = list(set(test_documents))[:3]
    print(f"📄 Found {len(test_documents)} test documents:")
    for doc in test_documents:
        print(f"   • {doc}")
    
    # Create output directory
    output_dir = Path('test_enhancement_results')
    output_dir.mkdir(exist_ok=True)
    
    # Test each document
    for i, doc_path in enumerate(test_documents):
        print(f"\n🔍 Testing document {i+1}/{len(test_documents)}: {doc_path.name}")
        
        try:
            # Check image dimensions first
            img = cv2.imread(str(doc_path))
            if img is None:
                print(f"⚠️ Cannot load image: {doc_path}")
                continue
            
            height, width = img.shape[:2]
            print(f"   📐 Dimensions: {width}x{height}")
            
            # Test both methods
            for method in ['line_detection']:  # Start with line_detection only
                print(f"   🔄 Testing {method} method...")
                
                output_path = output_dir / f"enhanced_{doc_path.stem}_{method}.png"
                
                # Run enhancement pipeline
                cmd = f"python full_document_enhancement.py --input {doc_path} --output {output_path} --model {latest_model} --method {method} --save-intermediates"
                
                print(f"   🚀 Running: {cmd}")
                exit_code = os.system(cmd)
                
                if exit_code == 0:
                    print(f"   ✅ Success! Output: {output_path}")
                else:
                    print(f"   ❌ Failed with exit code: {exit_code}")
                
        except Exception as e:
            print(f"   ❌ Error processing {doc_path}: {e}")
    
    print(f"\n🎯 Test Results")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   📊 Check comparison images for quality assessment")

def create_demo_document():
    """Create a synthetic demo document for testing"""
    print("🎨 Creating demo document...")
    
    # Create a simple synthetic document
    height, width = 1200, 800
    doc = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some text lines (simulated)
    lines = [
        "This is a test document for enhancement",
        "We are testing the full pipeline",
        "Including document preprocessing",
        "Line-by-line enhancement",
        "And document reconstruction",
        "This should work with the GAN-HTR model"
    ]
    
    y_start = 100
    line_height = 80
    
    for i, line in enumerate(lines):
        y = y_start + i * line_height
        cv2.putText(doc, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 0, 0), 2)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 10, doc.shape).astype(np.int16)
    doc = np.clip(doc.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Save demo document
    demo_path = "demo_document.png"
    cv2.imwrite(demo_path, doc)
    print(f"✅ Demo document created: {demo_path}")
    
    return demo_path

def quick_compatibility_test():
    """Quick test to verify all components work together"""
    print("⚡ Quick Compatibility Test")
    print("=" * 30)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from document_preprocessor import DocumentPreprocessor
        from enhance_document import DocumentEnhancer
        print("✅ All imports successful")
        
        # Test document preprocessor
        print("🔧 Testing document preprocessor...")
        preprocessor = DocumentPreprocessor()
        
        # Create a small test image
        test_img = np.ones((200, 800, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "Test line", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite("quick_test.png", test_img)
        
        # Test preprocessing
        segments = preprocessor.process_document("quick_test.png", method="line_detection")
        print(f"✅ Preprocessor generated {len(segments)} segments")
        
        # Test model loading
        print("🤖 Testing model loading...")
        model_files = list(Path('.').glob('generator_model_epoch_*.h5'))
        if model_files:
            latest_model = sorted(model_files)[-1]
            enhancer = DocumentEnhancer(str(latest_model))
            print("✅ Model loaded successfully")
        else:
            print("⚠️ No models found - skipping model test")
        
        # Cleanup
        os.remove("quick_test.png")
        
        print("🎉 All compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Full Document Enhancement - Test Suite")
    print("=" * 50)
    
    # Quick compatibility test first
    if not quick_compatibility_test():
        print("❌ Compatibility issues detected - fix these first")
        return
    
    print("\n" + "="*50)
    
    # Create demo document if no test documents available
    demo_created = False
    test_documents = []
    
    # Look for existing documents
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        test_documents.extend(list(Path('.').glob(ext)))
    
    if len(test_documents) < 2:
        print("📄 Creating demo document for testing...")
        demo_path = create_demo_document()
        demo_created = True
    
    # Run full tests
    test_with_sample_documents()
    
    # Cleanup demo if created
    if demo_created and os.path.exists("demo_document.png"):
        print("🧹 Cleaning up demo document...")
        os.remove("demo_document.png")
    
    print("\n🎯 Testing completed!")
    print("💡 Check 'test_enhancement_results/' for outputs")

if __name__ == "__main__":
    main()
