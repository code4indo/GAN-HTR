#!/usr/bin/env python3
"""
🚀 GAN-HTR Retraining with Fixed Dataset
========================================

Script untuk melatih ulang model GAN-HTR dengan dataset yang sudah
diperbaiki alignment-nya dan loss function yang ditingkatkan.

Author: Lambda One
Date: August 13, 2024
"""

import tensorflow as tf
import os
import numpy as np
import cv2
from pathlib import Path
import datetime
import glob
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Using {len(gpus)} GPUs")
    except RuntimeError as e:
        print(f"❌ GPU config error: {e}")

class ImprovedGANHTR:
    """Improved GAN-HTR with better loss functions and metrics"""
    
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Improved optimizers
        self.gen_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)
        self.disc_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
        
        # Multiple loss functions
        self.mse_loss = MeanSquaredError()
        self.bce_loss = BinaryCrossentropy(from_logits=True)
        
        # Metrics tracking
        self.train_metrics = {
            'gen_loss': [],
            'disc_loss': [],
            'psnr': [],
            'ssim': [],
            'pixel_accuracy': []
        }
    
    def build_generator(self):
        """Build enhanced U-Net generator"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder (downsampling)
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
        
        # Decoder (upsampling)
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
        
        # Output layer with sigmoid for better text enhancement
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_discriminator(self):
        """Build discriminator for adversarial training"""
        inputs = layers.Input(shape=self.input_shape)
        
        x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=x)
        return model
    
    def compute_psnr(self, y_true, y_pred):
        """Compute PSNR metric"""
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        psnr = 20 * tf.math.log(1.0) / tf.math.log(10.0) - 10 * tf.math.log(mse) / tf.math.log(10.0)
        return psnr
    
    def compute_ssim(self, y_true, y_pred):
        """Compute SSIM metric"""
        return tf.image.ssim(y_true, y_pred, max_val=1.0)
    
    def enhanced_generator_loss(self, disc_generated_output, gen_output, target):
        """Enhanced generator loss with multiple components"""
        # Adversarial loss
        gan_loss = self.bce_loss(tf.ones_like(disc_generated_output), disc_generated_output)
        
        # L1 loss for pixel-wise accuracy
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        
        # SSIM loss for structural similarity
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(target, gen_output, max_val=1.0))
        
        # Perceptual loss (simplified gradient-based)
        target_grad = tf.image.sobel_edges(target)
        gen_grad = tf.image.sobel_edges(gen_output)
        perceptual_loss = tf.reduce_mean(tf.abs(target_grad - gen_grad))
        
        # Combined loss
        total_gen_loss = gan_loss + (100 * l1_loss) + (10 * ssim_loss) + (5 * perceptual_loss)
        
        return total_gen_loss, gan_loss, l1_loss, ssim_loss, perceptual_loss
    
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """Discriminator loss"""
        real_loss = self.bce_loss(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.bce_loss(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss
    
    @tf.function
    def train_step(self, input_image, target_image):
        """Single training step"""
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)
            
            disc_real_output = self.discriminator(target_image, training=True)
            disc_generated_output = self.discriminator(gen_output, training=True)
            
            gen_total_loss, gen_gan_loss, gen_l1_loss, gen_ssim_loss, gen_perceptual_loss = \
                self.enhanced_generator_loss(disc_generated_output, gen_output, target_image)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
        
        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.gen_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
        
        # Calculate metrics
        psnr = self.compute_psnr(target_image, gen_output)
        ssim = tf.reduce_mean(self.compute_ssim(target_image, gen_output))
        
        return {
            'gen_total_loss': gen_total_loss,
            'gen_gan_loss': gen_gan_loss,
            'gen_l1_loss': gen_l1_loss,
            'gen_ssim_loss': gen_ssim_loss,
            'gen_perceptual_loss': gen_perceptual_loss,
            'disc_loss': disc_loss,
            'psnr': psnr,
            'ssim': ssim
        }

def load_aligned_dataset():
    """Load aligned dataset for training"""
    print("📂 Loading aligned dataset...")
    
    distorted_dir = "datasets/nan_aligned/train/distorted"
    gt_dir = "datasets/nan_aligned/train/gt"
    
    distorted_files = sorted(glob.glob(os.path.join(distorted_dir, "*.jpg")))
    
    inputs = []
    targets = []
    
    for dist_file in tqdm(distorted_files, desc="Loading training data"):
        filename = os.path.basename(dist_file)
        gt_file = os.path.join(gt_dir, filename)
        
        if os.path.exists(gt_file):
            try:
                # Load and preprocess
                dist_img = cv2.imread(dist_file, cv2.IMREAD_GRAYSCALE)
                gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
                
                if dist_img is not None and gt_img is not None:
                    # Resize to model input size
                    dist_resized = cv2.resize(dist_img, (128, 128))
                    gt_resized = cv2.resize(gt_img, (128, 128))
                    
                    # Normalize to [0, 1]
                    dist_norm = dist_resized.astype(np.float32) / 255.0
                    gt_norm = gt_resized.astype(np.float32) / 255.0
                    
                    # Add channel dimension
                    dist_norm = np.expand_dims(dist_norm, axis=-1)
                    gt_norm = np.expand_dims(gt_norm, axis=-1)
                    
                    inputs.append(dist_norm)
                    targets.append(gt_norm)
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"✅ Loaded {len(inputs)} training pairs")
    return np.array(inputs), np.array(targets)

def train_improved_model():
    """Train the improved GAN-HTR model"""
    print("🚀 STARTING IMPROVED GAN-HTR TRAINING")
    print("=" * 40)
    
    # Load aligned dataset
    X_train, y_train = load_aligned_dataset()
    
    if len(X_train) == 0:
        print("❌ No training data found! Run fix_dataset_alignment.py first")
        return
    
    print(f"📊 Training data shape: {X_train.shape}")
    print(f"📊 Target data shape: {y_train.shape}")
    
    # Initialize improved model
    model = ImprovedGANHTR(input_shape=(128, 128, 1))
    
    # Training parameters
    EPOCHS = 15
    BATCH_SIZE = 8
    steps_per_epoch = len(X_train) // BATCH_SIZE
    
    # Create timestamp for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoints/improved_model_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"💾 Checkpoints will be saved to: {checkpoint_dir}")
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"\n🔄 Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 30)
        
        epoch_metrics = {
            'gen_total_loss': [],
            'disc_loss': [],
            'psnr': [],
            'ssim': []
        }
        
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Training batches
        for step in tqdm(range(steps_per_epoch), desc=f"Training"):
            start_idx = step * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            
            batch_input = X_shuffled[start_idx:end_idx]
            batch_target = y_shuffled[start_idx:end_idx]
            
            # Train step
            metrics = model.train_step(batch_input, batch_target)
            
            # Collect metrics
            epoch_metrics['gen_total_loss'].append(float(metrics['gen_total_loss']))
            epoch_metrics['disc_loss'].append(float(metrics['disc_loss']))
            epoch_metrics['psnr'].append(float(metrics['psnr']))
            epoch_metrics['ssim'].append(float(metrics['ssim']))
        
        # Calculate epoch averages
        avg_gen_loss = np.mean(epoch_metrics['gen_total_loss'])
        avg_disc_loss = np.mean(epoch_metrics['disc_loss'])
        avg_psnr = np.mean(epoch_metrics['psnr'])
        avg_ssim = np.mean(epoch_metrics['ssim'])
        
        print(f"📊 Epoch {epoch + 1} Results:")
        print(f"   Generator Loss: {avg_gen_loss:.4f}")
        print(f"   Discriminator Loss: {avg_disc_loss:.4f}")
        print(f"   PSNR: {avg_psnr:.2f} dB")
        print(f"   SSIM: {avg_ssim:.4f}")
        
        # Save metrics
        model.train_metrics['gen_loss'].append(avg_gen_loss)
        model.train_metrics['disc_loss'].append(avg_disc_loss)
        model.train_metrics['psnr'].append(avg_psnr)
        model.train_metrics['ssim'].append(avg_ssim)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}")
            model.generator.save_weights(f"{checkpoint_path}_generator.weights.h5")
            model.discriminator.save_weights(f"{checkpoint_path}_discriminator.weights.h5")
            print(f"💾 Checkpoint saved: epoch {epoch + 1}")
        
        # Test enhancement quality on sample
        if (epoch + 1) % 3 == 0:
            test_enhancement_quality(model, epoch + 1)
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model")
    model.generator.save_weights(f"{final_model_path}_generator.weights.h5")
    model.discriminator.save_weights(f"{final_model_path}_discriminator.weights.h5")
    
    # Save training history
    save_training_history(model.train_metrics, checkpoint_dir)
    
    print(f"\n🎉 Training complete! Model saved to: {checkpoint_dir}")
    
    return model, checkpoint_dir

def test_enhancement_quality(model, epoch):
    """Test enhancement quality during training"""
    try:
        # Load test sample
        test_file = "datasets/nan_aligned/test/distorted/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg"
        gt_file = "datasets/nan_aligned/test/gt/001_NL-HaNA_1.04.02_8740_0147.tif_line_1679063660296_1211.jpg"
        
        if os.path.exists(test_file) and os.path.exists(gt_file):
            # Load and preprocess
            test_img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            
            test_resized = cv2.resize(test_img, (128, 128))
            gt_resized = cv2.resize(gt_img, (128, 128))
            
            test_norm = test_resized.astype(np.float32) / 255.0
            gt_norm = gt_resized.astype(np.float32) / 255.0
            
            test_input = np.expand_dims(np.expand_dims(test_norm, axis=-1), axis=0)
            gt_target = np.expand_dims(gt_norm, axis=-1)
            
            # Generate enhancement
            enhanced = model.generator(test_input, training=False)
            enhanced_np = enhanced[0].numpy()
            
            # Calculate metrics
            mse = np.mean((enhanced_np - gt_target) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            
            print(f"🧪 Epoch {epoch} test PSNR: {psnr:.2f} dB")
            
    except Exception as e:
        print(f"❌ Test error: {e}")

def save_training_history(metrics, checkpoint_dir):
    """Save training history plots"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(metrics['gen_loss'])
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 3, 2)
    plt.plot(metrics['disc_loss'])
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 3, 3)
    plt.plot(metrics['psnr'])
    plt.title('PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('dB')
    
    plt.subplot(2, 3, 4)
    plt.plot(metrics['ssim'])
    plt.title('SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    
    plt.subplot(2, 3, 5)
    plt.plot(metrics['gen_loss'], label='Generator')
    plt.plot(metrics['disc_loss'], label='Discriminator')
    plt.title('Combined Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training history saved to: {checkpoint_dir}/training_history.png")

def main():
    """Main training function"""
    print("🚀 IMPROVED GAN-HTR TRAINING WITH ALIGNED DATA")
    print("=" * 60)
    
    # Check if aligned dataset exists
    if not os.path.exists("datasets/nan_aligned"):
        print("❌ Aligned dataset not found!")
        print("Please run: python3 fix_dataset_alignment.py")
        return
    
    # Start training
    model, checkpoint_dir = train_improved_model()
    
    print("\n🎯 TRAINING COMPLETE!")
    print("=" * 25)
    print(f"Model saved to: {checkpoint_dir}")
    print("\nNext steps:")
    print("1. Test the trained model with enhanced test script")
    print("2. Compare results with previous model")
    print("3. Evaluate PSNR improvements")

if __name__ == "__main__":
    main()
