#!/usr/bin/env python3
"""
Test existing model enhancement with aligned dataset
"""

import os
import sys
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append('/home/lambda_one/tesis/GAN-HTR')

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    return ssim(img1, img2, data_range=255)

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess image for model input"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Resize to target size
    img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    return img

def load_image_for_display(image_path, target_size=(128, 128)):
    """Load image for display purposes"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Resize to target size
    img = cv2.resize(img, target_size)
    
    return img

def build_unet_generator(input_shape=(128, 128, 1)):
    """Build UNet Generator architecture"""
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Dropout, UpSampling2D, Concatenate
    from tensorflow.keras.models import Model
    
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv2D(64, 4, strides=2, padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    
    conv2 = Conv2D(128, 4, strides=2, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    
    conv3 = Conv2D(256, 4, strides=2, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    
    conv4 = Conv2D(512, 4, strides=2, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    
    conv5 = Conv2D(512, 4, strides=2, padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)
    
    conv6 = Conv2D(512, 4, strides=2, padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.2)(conv6)
    
    conv7 = Conv2D(512, 4, strides=2, padding='same')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.2)(conv7)
    
    # Bottleneck
    conv8 = Conv2D(512, 4, strides=2, padding='same')(conv7)
    conv8 = LeakyReLU(alpha=0.2)(conv8)
    
    # Decoder
    deconv1 = UpSampling2D(size=(2, 2))(conv8)
    deconv1 = Conv2D(512, 4, padding='same')(deconv1)
    deconv1 = BatchNormalization()(deconv1)
    deconv1 = Dropout(0.5)(deconv1)
    deconv1 = tf.nn.relu(deconv1)
    deconv1 = Concatenate()([deconv1, conv7])
    
    deconv2 = UpSampling2D(size=(2, 2))(deconv1)
    deconv2 = Conv2D(512, 4, padding='same')(deconv2)
    deconv2 = BatchNormalization()(deconv2)
    deconv2 = Dropout(0.5)(deconv2)
    deconv2 = tf.nn.relu(deconv2)
    deconv2 = Concatenate()([deconv2, conv6])
    
    deconv3 = UpSampling2D(size=(2, 2))(deconv2)
    deconv3 = Conv2D(512, 4, padding='same')(deconv3)
    deconv3 = BatchNormalization()(deconv3)
    deconv3 = Dropout(0.5)(deconv3)
    deconv3 = tf.nn.relu(deconv3)
    deconv3 = Concatenate()([deconv3, conv5])
    
    deconv4 = UpSampling2D(size=(2, 2))(deconv3)
    deconv4 = Conv2D(512, 4, padding='same')(deconv4)
    deconv4 = BatchNormalization()(deconv4)
    deconv4 = tf.nn.relu(deconv4)
    deconv4 = Concatenate()([deconv4, conv4])
    
    deconv5 = UpSampling2D(size=(2, 2))(deconv4)
    deconv5 = Conv2D(256, 4, padding='same')(deconv5)
    deconv5 = BatchNormalization()(deconv5)
    deconv5 = tf.nn.relu(deconv5)
    deconv5 = Concatenate()([deconv5, conv3])
    
    deconv6 = UpSampling2D(size=(2, 2))(deconv5)
    deconv6 = Conv2D(128, 4, padding='same')(deconv6)
    deconv6 = BatchNormalization()(deconv6)
    deconv6 = tf.nn.relu(deconv6)
    deconv6 = Concatenate()([deconv6, conv2])
    
    deconv7 = UpSampling2D(size=(2, 2))(deconv6)
    deconv7 = Conv2D(64, 4, padding='same')(deconv7)
    deconv7 = BatchNormalization()(deconv7)
    deconv7 = tf.nn.relu(deconv7)
    deconv7 = Concatenate()([deconv7, conv1])
    
    deconv8 = UpSampling2D(size=(2, 2))(deconv7)
    deconv8 = Conv2D(1, 4, padding='same')(deconv8)
    outputs = tf.nn.sigmoid(deconv8)
    
    model = Model(inputs, outputs)
    return model

def test_enhancement(distorted_path, ground_truth_path, model_weights_path, output_dir):
    """Test enhancement on a specific image pair"""
    
    print(f"🔍 Testing Enhancement Analysis")
    print(f"=====================================")
    print(f"📁 Distorted: {distorted_path}")
    print(f"📁 Ground truth: {ground_truth_path}")
    print(f"📁 Model weights: {model_weights_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"🔄 Loading model...")
    generator = build_unet_generator()
    
    try:
        generator.load_weights(model_weights_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Load images
    print(f"🔄 Loading images...")
    distorted_input = load_and_preprocess_image(distorted_path)
    ground_truth_img = load_image_for_display(ground_truth_path)
    distorted_display = load_image_for_display(distorted_path)
    
    if distorted_input is None or ground_truth_img is None:
        print(f"❌ Failed to load images")
        return None
    
    # Generate enhanced image
    print(f"🔄 Generating enhanced image...")
    enhanced_output = generator.predict(distorted_input, verbose=0)
    enhanced_img = (enhanced_output[0] * 255).astype(np.uint8).squeeze()
    
    # Calculate metrics
    print(f"📊 Calculating metrics...")
    
    # Baseline: Distorted vs Ground Truth
    baseline_psnr = calculate_psnr(distorted_display, ground_truth_img)
    baseline_ssim = calculate_ssim(distorted_display, ground_truth_img)
    
    # Enhanced: Enhanced vs Ground Truth
    enhanced_psnr = calculate_psnr(enhanced_img, ground_truth_img)
    enhanced_ssim = calculate_ssim(enhanced_img, ground_truth_img)
    
    # Print results
    print(f"\n📊 QUALITY METRICS")
    print(f"==================")
    print(f"Baseline (Distorted vs Ground Truth):")
    print(f"  PSNR: {baseline_psnr:.2f} dB")
    print(f"  SSIM: {baseline_ssim:.4f}")
    print(f"\nEnhanced (Enhanced vs Ground Truth):")
    print(f"  PSNR: {enhanced_psnr:.2f} dB")
    print(f"  SSIM: {enhanced_ssim:.4f}")
    print(f"\nImprovement:")
    print(f"  PSNR: {enhanced_psnr - baseline_psnr:+.2f} dB")
    print(f"  SSIM: {enhanced_ssim - baseline_ssim:+.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('GAN-HTR Enhancement Analysis', fontsize=16, fontweight='bold')
    
    # Top row: Images
    axes[0, 0].imshow(distorted_display, cmap='gray')
    axes[0, 0].set_title(f'Distorted Input\nPSNR: {baseline_psnr:.2f} dB, SSIM: {baseline_ssim:.4f}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(enhanced_img, cmap='gray')
    axes[0, 1].set_title(f'Enhanced Output\nPSNR: {enhanced_psnr:.2f} dB, SSIM: {enhanced_ssim:.4f}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(ground_truth_img, cmap='gray')
    axes[0, 2].set_title('Ground Truth')
    axes[0, 2].axis('off')
    
    # Bottom row: Difference maps
    diff_baseline = np.abs(distorted_display.astype(float) - ground_truth_img.astype(float))
    diff_enhanced = np.abs(enhanced_img.astype(float) - ground_truth_img.astype(float))
    diff_improvement = diff_baseline - diff_enhanced
    
    im1 = axes[1, 0].imshow(diff_baseline, cmap='hot')
    axes[1, 0].set_title('Baseline Error Map')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 1].imshow(diff_enhanced, cmap='hot')
    axes[1, 1].set_title('Enhanced Error Map')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    im3 = axes[1, 2].imshow(diff_improvement, cmap='RdBu', vmin=-50, vmax=50)
    axes[1, 2].set_title('Improvement Map\n(Blue=Better, Red=Worse)')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save results
    output_file = os.path.join(output_dir, f'enhancement_analysis_{Path(distorted_path).stem}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Results saved to: {output_file}")
    
    # Save individual images
    cv2.imwrite(os.path.join(output_dir, f'enhanced_{Path(distorted_path).stem}.png'), enhanced_img)
    
    plt.show()
    
    return {
        'baseline_psnr': baseline_psnr,
        'baseline_ssim': baseline_ssim,
        'enhanced_psnr': enhanced_psnr,
        'enhanced_ssim': enhanced_ssim,
        'psnr_improvement': enhanced_psnr - baseline_psnr,
        'ssim_improvement': enhanced_ssim - baseline_ssim
    }

def main():
    parser = argparse.ArgumentParser(description='Test GAN-HTR Enhancement with existing model')
    parser.add_argument('--distorted', required=True, help='Path to distorted image')
    parser.add_argument('--ground_truth', required=True, help='Path to ground truth image')
    parser.add_argument('--model_weights', required=True, help='Path to model weights')
    parser.add_argument('--output_dir', default='enhancement_test_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Test the enhancement
    results = test_enhancement(
        args.distorted,
        args.ground_truth,
        args.model_weights,
        args.output_dir
    )
    
    return results

if __name__ == "__main__":
    # Example usage with available files
    distorted_file = "datasets/nan_aligned/test/distorted/018_NL-HaNA_1.04.02_8740_0147.tif_r1l24.jpg"
    ground_truth_file = "datasets/nan_aligned/test/gt/018_NL-HaNA_1.04.02_8740_0147.tif_r1l24.jpg"
    model_weights = "ResultGanS_S_nan_OP/final/weights/generator.weights.h5"
    
    print("🚀 TESTING EXISTING MODEL WITH ALIGNED DATASET")
    print("=" * 50)
    
    if os.path.exists(distorted_file) and os.path.exists(ground_truth_file) and os.path.exists(model_weights):
        results = test_enhancement(
            distorted_file,
            ground_truth_file,
            model_weights,
            "enhancement_test_results"
        )
    else:
        print("❌ Files not found, checking available files...")
        # Check what files are available
        if os.path.exists("datasets/nan_aligned/test/distorted/"):
            distorted_files = list(Path("datasets/nan_aligned/test/distorted/").glob("*.jpg"))[:3]
            for f in distorted_files:
                gt_file = f"datasets/nan_aligned/test/gt/{f.name}"
                if os.path.exists(gt_file):
                    print(f"✅ Testing with: {f.name}")
                    results = test_enhancement(
                        str(f),
                        gt_file,
                        model_weights,
                        "enhancement_test_results"
                    )
                    break
