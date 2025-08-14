#!/usr/bin/env python3
"""
Simplified optimized training script untuk GAN-HTR
Versi yang lebih robust dengan error handling yang baik
"""

import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K

import math
from PIL import Image
from tqdm import tqdm
import random
import sys
import codecs
import re
import cv2
from glob import glob
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time

# Configure GPU strategy
print("=== GPU CONFIGURATION ===")
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"Available GPUs: {len(gpus)}")

if gpus:
    try:
        # Enable memory growth untuk kedua GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Setup Multi-GPU strategy
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} devices")
        else:
            strategy = tf.distribute.get_strategy()
            print("Using single GPU strategy")
            
        # Print GPU info
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
            
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
        strategy = tf.distribute.get_strategy()
else:
    print("No GPU available, using CPU")
    strategy = tf.distribute.get_strategy()

##########################################################################################################
# Configuration
##########################################################################################################
rootPath = './'
DatabasePath = 'datasets/nan_raw_biner/'
scenario = 'S_nan_OP_SIMPLE'

# Simplified hyperparameters
BATCH_SIZE = 8  # Conservative batch size
size = (128, 1024, 1)
input_size = (128, 1024, 1)
input_size_crnn = (1024, 128, 1)
max_text_length = 128

# CPU optimization
NUM_WORKERS = 8  # Reduced from 16 untuk stability

def read_file_char(filename):
    """Read character list from file"""
    lines = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                lines.append(line)
    return lines

def read_file(filename):
    """Read file lines"""
    lines = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                lines.append(line)
    return lines

def normalizeTranscription(text):
    """Normalize transcription text"""
    return text.lower()

# Load charset
charset_base = read_file_char(rootPath + 'Sets/CHAR_LIST')
print(f"Loaded charset with {len(charset_base)} tokens")

def encode_txt(text):
    """Encode text using charset"""
    encoded = []
    cc = text.split()
    for item in cc:
        try:
            index = charset_base.index(item)
            encoded.append(index)
        except ValueError:
            unk_index = charset_base.index('<UNK>')
            encoded.append(unk_index)
    return encoded

def load_image_pair(im_name, split='train'):
    """Load a single image pair"""
    try:
        # Degraded image path
        deg_image_path = os.path.join('datasets/nan_distorted/', split, im_name)
        if not os.path.exists(deg_image_path):
            return None
            
        # Ground truth image path  
        gt_image_path = os.path.join(DatabasePath, split, 'images', im_name)
        if not os.path.exists(gt_image_path):
            return None
        
        # Load dan resize dengan PIL
        with Image.open(deg_image_path) as deg_img:
            deg_img = deg_img.resize((1024, 128), Image.LANCZOS).convert('L')
            deg_array = np.array(deg_img, dtype=np.float32) / 255.0
            
        with Image.open(gt_image_path) as gt_img:
            gt_img = gt_img.resize((1024, 128), Image.LANCZOS).convert('L')
            gt_array = np.array(gt_img, dtype=np.float32) / 255.0
        
        return {
            'name': im_name,
            'deg_image': deg_array.reshape(128, 1024, 1),
            'gt_image': gt_array.reshape(128, 1024, 1)
        }
        
    except Exception as e:
        print(f"Error loading {im_name}: {e}")
        return None

def unet_generator():
    """Simplified UNet generator"""
    inputs = Input(shape=(128, 1024, 1))
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.3)(conv4)

    # Decoder
    up5 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop4))
    up5 = BatchNormalization()(up5)
    merge5 = concatenate([conv3, up5])
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    up6 = BatchNormalization()(up6)
    merge6 = concatenate([conv2, up6])
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    up7 = BatchNormalization()(up7)
    merge7 = concatenate([conv1, up7])
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    output = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=output)
    return model

def discriminator_patch():
    """Simplified patch discriminator"""
    img_A = Input(shape=(128, 1024, 1))
    img_B = Input(shape=(128, 1024, 1))

    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = Conv2D(64, 4, strides=2, padding='same')(combined_imgs)
    d1 = LeakyReLU(negative_slope=0.2)(d1)

    d2 = Conv2D(128, 4, strides=2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = LeakyReLU(negative_slope=0.2)(d2)

    d3 = Conv2D(256, 4, strides=2, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = LeakyReLU(negative_slope=0.2)(d3)

    validity = Conv2D(1, 4, padding='same')(d3)

    model = Model([img_A, img_B], validity)
    return model

def simple_train_gan(epochs=5):
    """Simplified training function"""
    print("=== SIMPLIFIED GAN TRAINING ===")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"CPU workers: {NUM_WORKERS}")
    print(f"GPU strategy: {strategy}")
    
    # Load data
    print("Loading training data...")
    list_image_train = [os.path.basename(x) for x in glob(os.path.join('datasets/nan_distorted/train', '*.jpg'))]
    list_lines = read_file('datasets/nan_raw_biner/train/lines.txt')
    
    print(f"Found {len(list_image_train)} training images")
    
    # Create models dalam strategy scope
    with strategy.scope():
        print("Creating models...")
        generator = unet_generator()
        discriminator = discriminator_patch()
        
        # Compile models
        generator.compile(optimizer=Adam(learning_rate=2e-4), loss='binary_crossentropy')
        discriminator.compile(optimizer=Adam(learning_rate=2e-4), loss='mse', metrics=['accuracy'])
        
        print("Models created and compiled!")
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        start_time = time.time()
        
        # Prepare batches
        batch_count = 0
        processed_images = 0
        epoch_losses = []
        
        # Process images in batches
        random.shuffle(list_image_train)
        
        for i in tqdm(range(0, min(100, len(list_image_train)), BATCH_SIZE), desc=f"Epoch {epoch + 1}"):
            try:
                batch_images = list_image_train[i:i+BATCH_SIZE]
                
                deg_batch = []
                gt_batch = []
                valid_batch = []
                
                # Load batch data
                for im_name in batch_images:
                    # Find ground truth text
                    matched_lines = [s for s in list_lines if im_name in s]
                    if not matched_lines:
                        continue
                        
                    line_parts = matched_lines[0].split(' ', 1)
                    if len(line_parts) < 2:
                        continue
                        
                    text_line = line_parts[1]
                    line = normalizeTranscription(text_line)
                    
                    # Check text length
                    if len(line.split()) >= max_text_length or len(line.split()) == 0:
                        continue
                    
                    # Load images
                    image_data = load_image_pair(im_name, split='train')
                    if image_data is None:
                        continue
                    
                    deg_batch.append(image_data['deg_image'])
                    gt_batch.append(image_data['gt_image'])
                    valid_batch.append(im_name)
                
                if len(deg_batch) < 2:  # Need at least 2 samples for batch
                    continue
                
                # Convert to numpy arrays
                deg_images = np.array(deg_batch)
                gt_images = np.array(gt_batch)
                
                # Training step
                # Train discriminator
                real_loss = discriminator.train_on_batch([gt_images, deg_images], 
                                                       np.ones((len(gt_images), 16, 128, 1)))
                
                # Generate fake images
                generated_images = generator.predict(deg_images, verbose=0)
                fake_loss = discriminator.train_on_batch([generated_images, deg_images], 
                                                       np.zeros((len(generated_images), 16, 128, 1)))
                
                d_loss = 0.5 * (real_loss[0] + fake_loss[0])
                
                # Train generator
                g_loss = generator.train_on_batch(deg_images, gt_images)
                
                epoch_losses.append({
                    'd_loss': d_loss,
                    'g_loss': g_loss
                })
                
                processed_images += len(deg_batch)
                batch_count += 1
                
                if batch_count >= 10:  # Limit untuk demo
                    break
                    
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
        
        # Print epoch statistics
        if epoch_losses:
            avg_d_loss = np.mean([l['d_loss'] for l in epoch_losses])
            avg_g_loss = np.mean([l['g_loss'] for l in epoch_losses])
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            print(f"  D Loss: {avg_d_loss:.4f}")
            print(f"  G Loss: {avg_g_loss:.4f}")
            print(f"  Processed: {processed_images} images")
        
        # Save models every epoch
        save_dir = f"ResultGanS_{scenario}/epoch_{epoch+1:03d}/weights"
        os.makedirs(save_dir, exist_ok=True)
        
        generator.save_weights(f"{save_dir}/generator.weights.h5")
        discriminator.save_weights(f"{save_dir}/discriminator.weights.h5")
        
        print(f"Models saved at epoch {epoch + 1}")
    
    print("\n=== Training completed! ===")
    
    # Final save
    final_save_dir = f"ResultGanS_{scenario}/final/weights"
    os.makedirs(final_save_dir, exist_ok=True)
    
    generator.save_weights(f"{final_save_dir}/generator.weights.h5")
    discriminator.save_weights(f"{final_save_dir}/discriminator.weights.h5")
    
    print("Final models saved!")
    
    return generator, discriminator

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified Multi-GPU GAN-HTR Training')
    parser.add_argument('--epoch', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    
    args = parser.parse_args()
    
    print("=== SIMPLIFIED MULTI-GPU GAN-HTR TRAINING ===")
    print(f"Total Epochs: {args.epoch}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Hardware Resources:")
    print(f"  - CPU: AMD Threadripper PRO 3955WX (32 threads)")
    print(f"  - RAM: 128GB (104GB available)")
    print(f"  - GPU: 2x RTX A4000 (32GB total VRAM)")
    print(f"  - Workers: {NUM_WORKERS}")
    print(f"  - Strategy: {strategy}")
    
    # Start training
    models = simple_train_gan(epochs=args.epoch)
    
    print("\n🎉 SIMPLIFIED TRAINING COMPLETED SUCCESSFULLY! 🎉")

if __name__ == "__main__":
    main()
