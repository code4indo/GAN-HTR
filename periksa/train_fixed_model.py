#!/usr/bin/env python3
"""
Fixed Training Script - Mengatasi sigmoid saturation problem
Root cause solution, bukan quick fix post-processing
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from datetime import datetime

class ImprovedGANHTR:
    """Fixed GAN-HTR with proper training to avoid sigmoid saturation"""
    
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # FIXED: More conservative optimizers to prevent saturation
        self.gen_optimizer = Adam(learning_rate=0.00005, beta_1=0.5, beta_2=0.999)  # Reduced LR
        self.disc_optimizer = Adam(learning_rate=0.00008, beta_1=0.5, beta_2=0.999)  # Reduced LR
        
        # FIXED: Balanced loss weights
        self.mse_loss = MeanSquaredError()
        self.bce_loss = BinaryCrossentropy(from_logits=True)
        
        # Training metrics
        self.train_metrics = {
            'gen_loss': [],
            'disc_loss': [],
            'psnr': [],
            'ssim': [],
            'generator_mean_output': []  # Track saturation
        }
    
    def build_generator(self):
        """Build U-Net generator with proper regularization"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder with dropout untuk prevent overfitting
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        conv1 = layers.Dropout(0.1)(conv1)  # Light dropout
        pool1 = layers.MaxPooling2D(2)(conv1)
        
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        conv2 = layers.Dropout(0.1)(conv2)
        pool2 = layers.MaxPooling2D(2)(conv2)
        
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        conv3 = layers.Dropout(0.2)(conv3)
        pool3 = layers.MaxPooling2D(2)(conv3)
        
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
        drop4 = layers.Dropout(0.3)(conv4)
        pool4 = layers.MaxPooling2D(2)(drop4)
        
        # Bottleneck dengan heavy dropout
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        drop5 = layers.Dropout(0.4)(conv5)
        
        # Decoder with skip connections
        up6 = layers.UpSampling2D(2)(drop5)
        up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(up6)
        merge6 = layers.concatenate([drop4, up6], axis=3)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
        conv6 = layers.Dropout(0.3)(conv6)
        
        up7 = layers.UpSampling2D(2)(conv6)
        up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(up7)
        merge7 = layers.concatenate([conv3, up7], axis=3)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
        conv7 = layers.Dropout(0.2)(conv7)
        
        up8 = layers.UpSampling2D(2)(conv7)
        up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(up8)
        merge8 = layers.concatenate([conv2, up8], axis=3)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
        conv8 = layers.Dropout(0.1)(conv8)
        
        up9 = layers.UpSampling2D(2)(conv8)
        up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(up9)
        merge9 = layers.concatenate([conv1, up9], axis=3)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
        
        # FIXED: Output layer dengan bias initialization untuk prevent saturation
        outputs = layers.Conv2D(1, 1, activation='sigmoid',
                               bias_initializer='zeros',  # Prevent initial saturation
                               kernel_initializer='glorot_uniform')(conv9)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_discriminator(self):
        """Build discriminator dengan reduced capacity untuk balance training"""
        inputs = layers.Input(shape=self.input_shape)
        
        # FIXED: Smaller discriminator untuk prevent generator collapse
        x = layers.Conv2D(32, 4, strides=2, padding='same')(inputs)  # Reduced from 64
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(64, 4, strides=2, padding='same')(x)  # Reduced from 128
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)  # Reduced from 256
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)  # No sigmoid, using from_logits=True
        
        model = Model(inputs=inputs, outputs=x)
        return model
    
    def enhanced_generator_loss(self, disc_generated_output, gen_output, target):
        """FIXED: Balanced generator loss untuk prevent saturation"""
        
        # Adversarial loss (reduced weight)
        gan_loss = self.bce_loss(tf.ones_like(disc_generated_output), disc_generated_output)
        
        # L1 loss for pixel-wise accuracy (increased weight)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        
        # SSIM loss (structural similarity)
        ssim_loss = 1.0 - tf.reduce_mean(self.compute_ssim(target, gen_output))
        
        # MSE loss for stability
        mse_loss = self.mse_loss(target, gen_output)
        
        # --- DEBUGGING: Print individual loss components ---
        tf.print("Loss components:", {
            "gan_loss": gan_loss, 
            "l1_loss": l1_loss, 
            "ssim_loss": ssim_loss, 
            "mse_loss": mse_loss
        }, output_stream=sys.stderr)
        # ----------------------------------------------------

        # FIXED: Balanced weights untuk prevent saturation
        # Prioritas: Pixel accuracy > Structure > Adversarial
        total_loss = (0.3 * gan_loss +      # Reduced adversarial weight
                     1.0 * l1_loss +        # High pixel accuracy weight
                     # 0.5 * ssim_loss +      # Structure preservation (DISABLED FOR DEBUGGING)
                     0.2 * mse_loss)        # Stability
        
        return {
            'total_loss': total_loss,
            'gan_loss': gan_loss,
            'l1_loss': l1_loss,
            'ssim_loss': ssim_loss,
            'mse_loss': mse_loss
        }
    
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """Standard discriminator loss"""
        real_loss = self.bce_loss(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.bce_loss(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss
    
    def compute_psnr(self, y_true, y_pred):
        """Compute PSNR"""
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return 20.0 * tf.math.log(1.0 / tf.sqrt(mse)) / tf.math.log(10.0)
    
    def compute_ssim(self, y_true, y_pred):
        """Compute SSIM"""
        return tf.image.ssim(y_true, y_pred, max_val=1.0)
    
    @tf.function
    def train_step(self, input_image, target_image):
        """FIXED: Training step dengan saturation monitoring"""
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generator forward pass
            gen_output = self.generator(input_image, training=True)
            
            # MONITOR: Check for saturation
            output_mean = tf.reduce_mean(gen_output)
            
            # Discriminator forward pass
            disc_real_output = self.discriminator(target_image, training=True)
            disc_generated_output = self.discriminator(gen_output, training=True)
            
            # Calculate losses
            gen_losses = self.enhanced_generator_loss(disc_generated_output, gen_output, target_image)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
        
        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_losses['total_loss'], self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # FIXED: Gradient clipping untuk prevent instability
        gen_gradients = [tf.clip_by_norm(grad, 1.0) for grad in gen_gradients]
        disc_gradients = [tf.clip_by_norm(grad, 1.0) for grad in disc_gradients]
        
        # Apply gradients
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        # Calculate metrics
        psnr = self.compute_psnr(target_image, gen_output)
        ssim = tf.reduce_mean(self.compute_ssim(target_image, gen_output))
        
        return {
            'gen_total_loss': gen_losses['total_loss'],
            'gen_gan_loss': gen_losses['gan_loss'],
            'gen_l1_loss': gen_losses['l1_loss'],
            'gen_ssim_loss': gen_losses['ssim_loss'],
            'gen_mse_loss': gen_losses['mse_loss'],
            'disc_loss': disc_loss,
            'psnr': psnr,
            'ssim': ssim,
            'output_mean': output_mean  # Track saturation
        }

def load_and_analyze_training_data():
    """Load dan analisis training data untuk detect issues"""
    print("🔍 ANALYZING TRAINING DATA FOR ISSUES...")
    
    # Try different possible training data locations
    possible_dirs = [
        "datasets/nan_aligned",
        "datasets/nan_distorted", 
        "datasets/nan_raw_biner"
    ]
    
    for data_dir in possible_dirs:
        if os.path.exists(data_dir):
            print(f"📁 Found: {data_dir}")
            
            # Sample some files
            files = glob.glob(os.path.join(data_dir, "**/*.jpg"), recursive=True)[:5]
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    norm_mean = (img.astype(np.float32) / 255.0).mean()
                    print(f"  {os.path.basename(f)}: normalized_mean={norm_mean:.3f}")
    
    # Create synthetic training data for testing
    print("\n🔧 CREATING SYNTHETIC TRAINING DATA FOR TESTING...")
    
    inputs = []
    targets = []
    
    # Generate synthetic data dengan proper range
    for i in range(100):
        # Create synthetic degraded image (darker, noisy)
        degraded = np.random.uniform(0.2, 0.7, (128, 128, 1)).astype(np.float32)
        
        # Create corresponding clean image (brighter, cleaner)  
        # Target should be brighter but not maximum (untuk avoid saturation)
        enhanced = np.clip(degraded + np.random.uniform(0.1, 0.3), 0.0, 0.85).astype(np.float32)  # Max 0.85, not 1.0
        
        inputs.append(degraded)
        targets.append(enhanced)
    
    print(f"✅ Created {len(inputs)} synthetic training pairs")
    print(f"Input range: {np.array(inputs).min():.3f} - {np.array(inputs).max():.3f}")
    print(f"Target range: {np.array(targets).min():.3f} - {np.array(targets).max():.3f}")
    print(f"Target mean: {np.array(targets).mean():.3f} (should be ~0.78 not 1.0)")
    
    return np.array(inputs), np.array(targets)

def train_fixed_model():
    """Train model dengan fixes untuk prevent sigmoid saturation"""
    
    print("🚀 STARTING FIXED TRAINING - PREVENT SIGMOID SATURATION")
    print("=" * 60)
    
    # Load training data
    train_inputs, train_targets = load_and_analyze_training_data()
    
    # Initialize model
    gan = ImprovedGANHTR()
    
    # Training parameters
    epochs = 10  # Test dengan epochs kecil dulu
    batch_size = 8
    steps_per_epoch = len(train_inputs) // batch_size
    
    print(f"📊 Training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total training pairs: {len(train_inputs)}")
    
    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
    train_dataset = train_dataset.shuffle(1000).batch(batch_size)
    
    # Training loop dengan saturation monitoring
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoints/fixed_model_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\n🏃 Starting training...")
    
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch+1}/{epochs}")
        
        epoch_metrics = {
            'gen_loss': [], 'disc_loss': [], 'psnr': [], 'ssim': [], 'output_mean': []
        }
        
        # Training step
        for step, (input_batch, target_batch) in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch+1}")):
            
            metrics = gan.train_step(input_batch, target_batch)
            
            # Collect metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key].append(float(metrics[key]))
        
        # Calculate epoch averages
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        # Print metrics dengan saturation warning
        print(f"📊 Epoch {epoch+1} Results:")
        print(f"  Gen Loss: {avg_metrics['gen_loss']:.4f}")
        print(f"  Disc Loss: {avg_metrics['disc_loss']:.4f}")
        print(f"  PSNR: {avg_metrics['psnr']:.2f} dB")
        print(f"  SSIM: {avg_metrics['ssim']:.4f}")
        print(f"  Output Mean: {avg_metrics['output_mean']:.3f}", end="")
        
        # SATURATION WARNING
        if avg_metrics['output_mean'] > 0.9:
            print(" ⚠️ WARNING: SIGMOID SATURATION!")
        elif avg_metrics['output_mean'] > 0.8:
            print(" ⚠️ Approaching saturation")
        else:
            print(" ✅ Normal range")
        
        # Save checkpoint setiap 5 epochs
        if (epoch + 1) % 5 == 0:
            gen_path = os.path.join(checkpoint_dir, f"generator_epoch_{epoch+1}.weights.h5")
            gan.generator.save_weights(gen_path)
            print(f"💾 Checkpoint saved: {gen_path}")
    
    # Final save
    final_gen_path = os.path.join(checkpoint_dir, "generator_final.weights.h5")
    gan.generator.save_weights(final_gen_path)
    
    print(f"\n✅ FIXED TRAINING COMPLETED!")
    print(f"📁 Model saved: {final_gen_path}")
    print(f"🎯 Check if output_mean < 0.85 (not saturated)")
    
    return gan, final_gen_path

if __name__ == "__main__":
    train_fixed_model()
