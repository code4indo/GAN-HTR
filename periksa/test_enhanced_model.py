#!/usr/bin/env python3
"""
🧪 Enhanced Model Testing & Validation
======================================

Script untuk menguji model yang sudah diperbaiki dan membandingkan
hasil enhancement sebelum dan sesudah perbaikan dataset.

Author: Lambda One
Date: August 13, 2024
"""

import tensorflow as tf
import cv2
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Import model architecture
import sys
sys.path.append('.')

class ModelTester:
    """Enhanced model testing and validation"""
    
    def __init__(self):
        self.input_shape = (128, 128, 1)
        self.generator = None
        
    def load_old_model(self):
        """Load the old model for comparison"""
        try:
            # Load old model architecture
            from network.model import build_generator
            
            old_generator = build_generator()
            
            # Find latest checkpoint
            checkpoint_files = glob.glob("checkpoints/*/generator_*.h5")
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                old_generator.load_weights(latest_checkpoint)
                print(f"✅ Loaded old model from: {latest_checkpoint}")
                return old_generator
            else:
                print("❌ No old model checkpoints found")
                return None
                
        except Exception as e:
            print(f"❌ Error loading old model: {e}")
            return None
    
    def build_new_generator(self):
        """Build new improved generator architecture"""
        from tensorflow.keras import layers, Model
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder
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
        
        # Decoder
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
        
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
        
        return Model(inputs=inputs, outputs=outputs)
    
    def load_new_model(self):
        """Load the new improved model"""
        try:
            new_generator = self.build_new_generator()
            
            # Find latest improved model checkpoint
            improved_checkpoints = glob.glob("checkpoints/improved_model_*/final_model_generator.h5")
            if improved_checkpoints:
                latest_improved = max(improved_checkpoints, key=os.path.getctime)
                new_generator.load_weights(latest_improved)
                print(f"✅ Loaded new model from: {latest_improved}")
                return new_generator
            else:
                print("❌ No improved model checkpoints found")
                return None
                
        except Exception as e:
            print(f"❌ Error loading new model: {e}")
            return None
    
    def calculate_metrics(self, original, enhanced, ground_truth):
        """Calculate comprehensive metrics"""
        # Convert to proper range for metrics
        original = np.clip(original * 255, 0, 255).astype(np.uint8)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        ground_truth = np.clip(ground_truth * 255, 0, 255).astype(np.uint8)
        
        # PSNR
        mse_orig = np.mean((original.astype(float) - ground_truth.astype(float)) ** 2)
        mse_enh = np.mean((enhanced.astype(float) - ground_truth.astype(float)) ** 2)
        
        psnr_orig = 20 * np.log10(255.0 / np.sqrt(mse_orig)) if mse_orig > 0 else float('inf')
        psnr_enh = 20 * np.log10(255.0 / np.sqrt(mse_enh)) if mse_enh > 0 else float('inf')
        
        # SSIM
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if len(enhanced.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        if len(ground_truth.shape) == 3:
            ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
        
        ssim_orig = ssim(ground_truth, original, data_range=255)
        ssim_enh = ssim(ground_truth, enhanced, data_range=255)
        
        # Improvement metrics
        psnr_improvement = psnr_enh - psnr_orig
        ssim_improvement = ssim_enh - ssim_orig
        
        return {
            'psnr_original': psnr_orig,
            'psnr_enhanced': psnr_enh,
            'psnr_improvement': psnr_improvement,
            'ssim_original': ssim_orig,
            'ssim_enhanced': ssim_enh,
            'ssim_improvement': ssim_improvement
        }
    
    def test_single_image(self, image_path, ground_truth_path, old_model, new_model):
        """Test enhancement on a single image"""
        try:
            # Load images
            distorted = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            
            if distorted is None or ground_truth is None:
                return None
            
            # Preprocess for model input
            distorted_resized = cv2.resize(distorted, (128, 128))
            gt_resized = cv2.resize(ground_truth, (128, 128))
            
            distorted_norm = distorted_resized.astype(np.float32) / 255.0
            gt_norm = gt_resized.astype(np.float32) / 255.0
            
            model_input = np.expand_dims(np.expand_dims(distorted_norm, axis=-1), axis=0)
            
            results = {}
            
            # Test old model if available
            if old_model is not None:
                try:
                    old_enhanced = old_model(model_input, training=False)
                    old_enhanced_np = old_enhanced[0].numpy().squeeze()
                    
                    old_metrics = self.calculate_metrics(distorted_norm, old_enhanced_np, gt_norm)
                    results['old'] = {
                        'enhanced': old_enhanced_np,
                        'metrics': old_metrics
                    }
                except Exception as e:
                    print(f"❌ Old model test failed: {e}")
            
            # Test new model if available
            if new_model is not None:
                try:
                    new_enhanced = new_model(model_input, training=False)
                    new_enhanced_np = new_enhanced[0].numpy().squeeze()
                    
                    new_metrics = self.calculate_metrics(distorted_norm, new_enhanced_np, gt_norm)
                    results['new'] = {
                        'enhanced': new_enhanced_np,
                        'metrics': new_metrics
                    }
                except Exception as e:
                    print(f"❌ New model test failed: {e}")
            
            # Add original data for comparison
            results['original'] = {
                'distorted': distorted_norm,
                'ground_truth': gt_norm
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error testing image {image_path}: {e}")
            return None
    
    def compare_models(self):
        """Compare old and new models on test dataset"""
        print("🔬 COMPARING OLD VS NEW MODELS")
        print("=" * 40)
        
        # Load models
        old_model = self.load_old_model()
        new_model = self.load_new_model()
        
        if old_model is None and new_model is None:
            print("❌ No models available for testing")
            return
        
        # Get test files
        test_distorted_dir = "datasets/nan_aligned/test/distorted"
        test_gt_dir = "datasets/nan_aligned/test/gt"
        
        if not os.path.exists(test_distorted_dir):
            print("❌ Aligned test dataset not found")
            print("Please run: python3 fix_dataset_alignment.py")
            return
        
        test_files = glob.glob(os.path.join(test_distorted_dir, "*.jpg"))[:5]  # Test first 5 files
        
        all_results = []
        
        for test_file in tqdm(test_files, desc="Testing images"):
            filename = os.path.basename(test_file)
            gt_file = os.path.join(test_gt_dir, filename)
            
            if os.path.exists(gt_file):
                results = self.test_single_image(test_file, gt_file, old_model, new_model)
                if results:
                    results['filename'] = filename
                    all_results.append(results)
        
        # Analyze results
        self.analyze_comparison_results(all_results)
        
        # Create visualization
        self.create_comparison_visualization(all_results)
    
    def analyze_comparison_results(self, results):
        """Analyze and print comparison results"""
        print("\n📊 COMPARISON RESULTS")
        print("=" * 25)
        
        old_psnr_improvements = []
        new_psnr_improvements = []
        old_ssim_improvements = []
        new_ssim_improvements = []
        
        for result in results:
            filename = result['filename']
            print(f"\n📄 {filename}")
            print("-" * 50)
            
            if 'old' in result:
                old_metrics = result['old']['metrics']
                old_psnr_improvements.append(old_metrics['psnr_improvement'])
                old_ssim_improvements.append(old_metrics['ssim_improvement'])
                print(f"🔵 Old Model:")
                print(f"   PSNR: {old_metrics['psnr_original']:.2f} → {old_metrics['psnr_enhanced']:.2f} dB (Δ{old_metrics['psnr_improvement']:+.2f})")
                print(f"   SSIM: {old_metrics['ssim_original']:.4f} → {old_metrics['ssim_enhanced']:.4f} (Δ{old_metrics['ssim_improvement']:+.4f})")
            
            if 'new' in result:
                new_metrics = result['new']['metrics']
                new_psnr_improvements.append(new_metrics['psnr_improvement'])
                new_ssim_improvements.append(new_metrics['ssim_improvement'])
                print(f"🟢 New Model:")
                print(f"   PSNR: {new_metrics['psnr_original']:.2f} → {new_metrics['psnr_enhanced']:.2f} dB (Δ{new_metrics['psnr_improvement']:+.2f})")
                print(f"   SSIM: {new_metrics['ssim_original']:.4f} → {new_metrics['ssim_enhanced']:.4f} (Δ{new_metrics['ssim_improvement']:+.4f})")
        
        # Summary statistics
        print(f"\n🎯 SUMMARY STATISTICS")
        print("=" * 25)
        
        if old_psnr_improvements:
            print(f"🔵 Old Model Average:")
            print(f"   PSNR Improvement: {np.mean(old_psnr_improvements):+.2f} ± {np.std(old_psnr_improvements):.2f} dB")
            print(f"   SSIM Improvement: {np.mean(old_ssim_improvements):+.4f} ± {np.std(old_ssim_improvements):.4f}")
        
        if new_psnr_improvements:
            print(f"🟢 New Model Average:")
            print(f"   PSNR Improvement: {np.mean(new_psnr_improvements):+.2f} ± {np.std(new_psnr_improvements):.2f} dB")
            print(f"   SSIM Improvement: {np.mean(new_ssim_improvements):+.4f} ± {np.std(new_ssim_improvements):.4f}")
        
        if old_psnr_improvements and new_psnr_improvements:
            psnr_diff = np.mean(new_psnr_improvements) - np.mean(old_psnr_improvements)
            ssim_diff = np.mean(new_ssim_improvements) - np.mean(old_ssim_improvements)
            
            print(f"\n🚀 IMPROVEMENT (New vs Old):")
            print(f"   PSNR: {psnr_diff:+.2f} dB better")
            print(f"   SSIM: {ssim_diff:+.4f} better")
            
            if psnr_diff > 0:
                print("✅ New model shows improvement!")
            else:
                print("⚠️ New model needs further optimization")
    
    def create_comparison_visualization(self, results):
        """Create visual comparison of results"""
        if not results:
            return
        
        # Take first result for detailed visualization
        result = results[0]
        filename = result['filename']
        
        plt.figure(figsize=(20, 15))
        
        # Prepare images
        original_dist = result['original']['distorted']
        original_gt = result['original']['ground_truth']
        
        num_cols = 3
        if 'old' in result:
            num_cols += 1
        if 'new' in result:
            num_cols += 1
        
        col = 1
        
        # Original distorted
        plt.subplot(3, num_cols, col)
        plt.imshow(original_dist, cmap='gray')
        plt.title('Original Distorted')
        plt.axis('off')
        col += 1
        
        # Ground truth
        plt.subplot(3, num_cols, col)
        plt.imshow(original_gt, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        col += 1
        
        # Old model result
        if 'old' in result:
            plt.subplot(3, num_cols, col)
            plt.imshow(result['old']['enhanced'], cmap='gray')
            old_psnr = result['old']['metrics']['psnr_improvement']
            plt.title(f'Old Model\n(PSNR: {old_psnr:+.2f} dB)')
            plt.axis('off')
            col += 1
        
        # New model result
        if 'new' in result:
            plt.subplot(3, num_cols, col)
            plt.imshow(result['new']['enhanced'], cmap='gray')
            new_psnr = result['new']['metrics']['psnr_improvement']
            plt.title(f'New Model\n(PSNR: {new_psnr:+.2f} dB)')
            plt.axis('off')
            col += 1
        
        # Difference maps (if both models available)
        if 'old' in result and 'new' in result:
            # Old model difference
            plt.subplot(3, num_cols, num_cols + 1)
            old_diff = np.abs(result['old']['enhanced'] - original_gt)
            plt.imshow(old_diff, cmap='hot')
            plt.title('Old Model\nDifference Map')
            plt.axis('off')
            
            # New model difference
            plt.subplot(3, num_cols, num_cols + 2)
            new_diff = np.abs(result['new']['enhanced'] - original_gt)
            plt.imshow(new_diff, cmap='hot')
            plt.title('New Model\nDifference Map')
            plt.axis('off')
        
        plt.suptitle(f'Model Comparison: {filename}', fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'comparison_{filename}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison visualization saved to: test_results/comparison_{filename}.png")

def test_specific_problematic_file():
    """Test the specific file that was problematic before"""
    print("🎯 TESTING SPECIFIC PROBLEMATIC FILE")
    print("=" * 40)
    
    tester = ModelTester()
    
    # Test the specific file that showed poor results
    test_file = "datasets/nan_aligned/test/distorted/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg"
    gt_file = "datasets/nan_aligned/test/gt/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg"
    
    if not os.path.exists(test_file) or not os.path.exists(gt_file):
        print("❌ Test files not found. Please run fix_dataset_alignment.py first")
        return
    
    # Load models
    old_model = tester.load_old_model()
    new_model = tester.load_new_model()
    
    # Test the specific file
    results = tester.test_single_image(test_file, gt_file, old_model, new_model)
    
    if results:
        print("\n📊 SPECIFIC FILE TEST RESULTS")
        print("=" * 35)
        
        if 'old' in results:
            old_metrics = results['old']['metrics']
            print(f"🔵 Old Model:")
            print(f"   PSNR Improvement: {old_metrics['psnr_improvement']:+.2f} dB")
            print(f"   SSIM Improvement: {old_metrics['ssim_improvement']:+.4f}")
        
        if 'new' in results:
            new_metrics = results['new']['metrics']
            print(f"🟢 New Model:")
            print(f"   PSNR Improvement: {new_metrics['psnr_improvement']:+.2f} dB")
            print(f"   SSIM Improvement: {new_metrics['ssim_improvement']:+.4f}")
        
        # Create detailed visualization for this specific file
        tester.create_comparison_visualization([{**results, 'filename': 'problematic_test_file.jpg'}])

def main():
    """Main testing function"""
    print("🧪 ENHANCED MODEL TESTING & VALIDATION")
    print("=" * 50)
    
    # Check requirements
    if not os.path.exists("datasets/nan_aligned"):
        print("❌ Aligned dataset not found!")
        print("Please run: python3 fix_dataset_alignment.py")
        return
    
    tester = ModelTester()
    
    print("\nChoose testing option:")
    print("1. Compare all models on test dataset")
    print("2. Test specific problematic file")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        tester.compare_models()
    
    if choice in ['2', '3']:
        test_specific_problematic_file()
    
    print("\n✅ Testing complete!")
    print("Check test_results/ directory for visualizations")

if __name__ == "__main__":
    main()
