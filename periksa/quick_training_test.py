#!/usr/bin/env python3
"""
Quick training test - shortened version for validation
"""

import os
import sys
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import datetime

# Add project root to path
sys.path.append('/home/lambda_one/tesis/GAN-HTR')

# Set up GPU strategy
print("🔧 Setting up GPU strategy...")
strategy = tf.distribute.MirroredStrategy()
print(f"✅ Using {strategy.num_replicas_in_sync} GPU(s)")

def load_and_preprocess_data(max_samples=100):
    """Load and preprocess a small subset of data for testing"""
    
    print(f"📂 Loading test dataset (max {max_samples} samples)...")
    
    train_distorted_dir = "datasets/nan_aligned/train/distorted"
    train_gt_dir = "datasets/nan_aligned/train/gt"
    
    distorted_files = sorted(list(Path(train_distorted_dir).glob("*.jpg")))[:max_samples]
    
    X_train = []
    y_train = []
    
    for file_path in tqdm(distorted_files, desc="Loading images"):
        gt_path = os.path.join(train_gt_dir, file_path.name)
        
        if os.path.exists(gt_path):
            # Load distorted image
            img_dist = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            if img_dist is not None and img_gt is not None:
                # Resize to 128x128
                img_dist = cv2.resize(img_dist, (128, 128))
                img_gt = cv2.resize(img_gt, (128, 128))
                
                # Normalize to [0, 1]
                img_dist = img_dist.astype(np.float32) / 255.0
                img_gt = img_gt.astype(np.float32) / 255.0
                
                # Add channel dimension
                img_dist = np.expand_dims(img_dist, axis=-1)
                img_gt = np.expand_dims(img_gt, axis=-1)
                
                X_train.append(img_dist)
                y_train.append(img_gt)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"✅ Loaded {len(X_train)} training pairs")
    print(f"📊 Training data shape: {X_train.shape}")
    print(f"📊 Target data shape: {y_train.shape}")
    
    return X_train, y_train

def build_simple_generator():
    """Build a simplified generator for testing"""
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Conv2DTranspose, ReLU, Activation
    from tensorflow.keras.models import Model
    
    inputs = Input(shape=(128, 128, 1))
    
    # Encoder
    x = Conv2D(32, 4, strides=2, padding='same')(inputs)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    x = Conv2D(64, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    # Decoder
    x = Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2DTranspose(32, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2DTranspose(1, 4, strides=2, padding='same')(x)
    outputs = Activation('sigmoid')(x)
    
    model = Model(inputs, outputs, name='simple_generator')
    return model

def build_simple_discriminator():
    """Build a simplified discriminator for testing"""
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Flatten, Dense
    from tensorflow.keras.models import Model
    
    inputs = Input(shape=(128, 128, 1))
    
    x = Conv2D(32, 4, strides=2, padding='same')(inputs)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    x = Conv2D(64, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    x = Flatten()(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs, name='simple_discriminator')
    return model

def calculate_psnr(y_true, y_pred):
    """Calculate PSNR"""
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return 20 * tf.math.log(1.0 / tf.sqrt(mse)) / tf.math.log(10.0)

@tf.function
def train_step(generator, discriminator, gen_optimizer, disc_optimizer, real_images, distorted_images):
    """Training step function"""
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate enhanced images
        enhanced_images = generator(distorted_images, training=True)
        
        # Discriminator predictions
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(enhanced_images, training=True)
        
        # Generator loss (MSE + adversarial)
        mse_loss = tf.reduce_mean(tf.square(real_images - enhanced_images))
        gen_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(fake_output), logits=fake_output))
        gen_loss = mse_loss * 100 + gen_adv_loss
        
        # Discriminator loss
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(real_output), logits=real_output))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(fake_output), logits=fake_output))
        disc_loss = real_loss + fake_loss
    
    # Calculate gradients
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply gradients
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    # Calculate PSNR
    psnr = calculate_psnr(real_images, enhanced_images)
    
    return gen_loss, disc_loss, psnr

def quick_training_test():
    """Run a quick training test"""
    
    print("🚀 QUICK TRAINING TEST")
    print("=" * 30)
    
    # Load small dataset
    X_train, y_train = load_and_preprocess_data(max_samples=50)
    
    with strategy.scope():
        # Build models
        print("🏗️ Building models...")
        generator = build_simple_generator()
        discriminator = build_simple_discriminator()
        
        # Optimizers
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        print("✅ Models built successfully")
        
        # Create dataset
        batch_size = 4
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(batch_size)
        dataset = strategy.experimental_distribute_dataset(dataset)
        
        print(f"📦 Created batched dataset with batch size {batch_size}")
        
        # Training loop
        print("\n🔄 Starting training test (2 epochs)...")
        
        for epoch in range(2):
            print(f"\nEpoch {epoch + 1}/2")
            print("-" * 20)
            
            epoch_gen_loss = []
            epoch_disc_loss = []
            epoch_psnr = []
            
            batch_count = 0
            for batch_data in dataset:
                distorted_batch, real_batch = batch_data
                
                # Distributed training step
                per_replica_losses = strategy.run(
                    train_step, 
                    args=(generator, discriminator, gen_optimizer, disc_optimizer, real_batch, distorted_batch)
                )
                
                # Reduce losses
                gen_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[0], axis=None)
                disc_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[1], axis=None)
                psnr = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[2], axis=None)
                
                epoch_gen_loss.append(gen_loss.numpy())
                epoch_disc_loss.append(disc_loss.numpy())
                epoch_psnr.append(psnr.numpy())
                
                batch_count += 1
                if batch_count % 5 == 0:
                    print(f"  Batch {batch_count}: Gen Loss: {gen_loss:.4f}, "
                          f"Disc Loss: {disc_loss:.4f}, PSNR: {psnr:.2f} dB")
            
            # Epoch summary
            avg_gen_loss = np.mean(epoch_gen_loss)
            avg_disc_loss = np.mean(epoch_disc_loss)
            avg_psnr = np.mean(epoch_psnr)
            
            print(f"📊 Epoch {epoch + 1} Results:")
            print(f"   Generator Loss: {avg_gen_loss:.4f}")
            print(f"   Discriminator Loss: {avg_disc_loss:.4f}")
            print(f"   PSNR: {avg_psnr:.2f} dB")
    
    print("\n✅ TRAINING TEST COMPLETED SUCCESSFULLY!")
    print("🎯 Ready for full training with train_improved_model.py")

if __name__ == "__main__":
    quick_training_test()
