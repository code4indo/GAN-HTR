#!/usr/bin/env python3
"""
Multi-GPU optimized training script untuk GAN-HTR
Memanfaatkan seluruh resource hardware secara maksimal
"""

import os
# Set environment untuk multi-GPU
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # Gunakan kedua GPU
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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
        strategy = tf.distribute.MirroredStrategy()
        print(f"Number of devices: {strategy.num_replicas_in_sync}")
        
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
# Optimized Configuration
##########################################################################################################
rootPath = './'
DatabasePath = 'datasets/nan_raw_biner/'
scenario = 'S_nan_OP_OPTIMIZED'

# Optimized hyperparameters untuk hardware ini
GLOBAL_BATCH_SIZE = 32  # Increased batch size
PER_REPLICA_BATCH_SIZE = GLOBAL_BATCH_SIZE // strategy.num_replicas_in_sync

size = (128, 1024, 1)
input_size = (128, 1024, 1)
input_size_crnn = (1024, 128, 1)
max_text_length = 128

# CPU optimization
NUM_CPU_CORES = 32
NUM_WORKERS = min(16, NUM_CPU_CORES - 4)  # Reserve 4 cores for system
PREFETCH_BUFFER = tf.data.AUTOTUNE

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

def parallel_image_loader(image_paths, split='train', num_workers=NUM_WORKERS):
    """Parallel image loading menggunakan ThreadPoolExecutor"""
    def load_single_image(im_path):
        try:
            im_name = os.path.basename(im_path)
            
            # Degraded image
            deg_image_path = os.path.join('datasets/nan_distorted/', split, im_name)
            if not os.path.exists(deg_image_path):
                return None
                
            # Ground truth image  
            gt_image_path = os.path.join(DatabasePath, split, 'images', im_name)
            if not os.path.exists(gt_image_path):
                return None
            
            # Load dan resize dengan PIL (lebih cepat)
            with Image.open(deg_image_path) as deg_img:
                deg_img = deg_img.resize((1024, 128), Image.LANCZOS).convert('L')
                deg_array = np.array(deg_img, dtype=np.float32) / 255.0
                
            with Image.open(gt_image_path) as gt_img:
                gt_img = gt_img.resize((1024, 128), Image.LANCZOS).convert('L')
                gt_array = np.array(gt_img, dtype=np.float32) / 255.0
            
            return {
                'name': im_name,
                'deg_image': deg_array,
                'gt_image': gt_array
            }
            
        except Exception as e:
            print(f"Error loading {im_path}: {e}")
            return None
    
    # Parallel loading
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(load_single_image, image_paths),
            total=len(image_paths),
            desc=f"Loading {split} images"
        ))
    
    # Filter None results
    return [r for r in results if r is not None]

def create_optimized_dataset(image_data, text_data, batch_size):
    """Create optimized tf.data.Dataset"""
    
    def generator():
        for img_data in image_data:
            im_name = img_data['name']
            
            # Find matching text
            matched_lines = [s for s in text_data if im_name in s]
            if not matched_lines:
                continue
                
            line_parts = matched_lines[0].split(' ', 1)
            if len(line_parts) < 2:
                continue
                
            text_line = line_parts[1]
            line = normalizeTranscription(text_line)
            
            if len(line.split()) >= max_text_length or len(line.split()) == 0:
                continue
            
            # Encode text
            encoded_text = encode_txt(line)
            
            # Pad text
            padded_text = encoded_text + [0] * (max_text_length - len(encoded_text))
            padded_text = padded_text[:max_text_length]
            
            yield (
                img_data['deg_image'].reshape(128, 1024, 1),
                img_data['gt_image'].reshape(128, 1024, 1), 
                np.array(padded_text, dtype=np.int32)
            )
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(max_text_length,), dtype=tf.int32)
        )
    )
    
    # Optimize dataset pipeline
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(PREFETCH_BUFFER)
    
    return dataset

def unet_generator():
    """Optimized UNet generator"""
    inputs = Input(shape=(128, 1024, 1))
    
    # Encoder dengan batch normalization dan dropout
    conv1 = Conv2D(64, 3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.3)(conv4)  # Reduced dropout
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.3)(conv5)

    # Decoder
    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    up6 = BatchNormalization()(up6)
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    up7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    
    output = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=output)
    return model

def discriminator_patch():
    """Optimized patch discriminator"""
    img_A = Input(shape=(128, 1024, 1))
    img_B = Input(shape=(128, 1024, 1))

    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = Conv2D(64, 4, strides=2, padding='same')(combined_imgs)
    d1 = LeakyReLU(negative_slope=0.2)(d1)

    d2 = Conv2D(128, 4, strides=2, padding='same')(d1)
    d2 = BatchNormalization(momentum=0.8)(d2)
    d2 = LeakyReLU(negative_slope=0.2)(d2)

    d3 = Conv2D(256, 4, strides=2, padding='same')(d2)
    d3 = BatchNormalization(momentum=0.8)(d3)
    d3 = LeakyReLU(negative_slope=0.2)(d3)

    d4 = Conv2D(512, 4, padding='same')(d3)
    d4 = BatchNormalization(momentum=0.8)(d4)
    d4 = LeakyReLU(negative_slope=0.2)(d4)

    validity = Conv2D(1, 4, padding='same')(d4)

    model = Model([img_A, img_B], validity)
    return model

def ctc_loss_lambda_func(y_true, y_pred):
    """Optimized CTC loss"""
    if len(y_true.shape) > 2:
        y_true = tf.squeeze(y_true)

    input_length = tf.math.reduce_sum(y_pred, axis=2, keepdims=False)
    input_length = tf.math.reduce_sum(input_length, axis=1, keepdims=True)

    label_length = tf.math.reduce_sum(tf.cast(y_true, tf.float32), axis=1, keepdims=True)

    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)

    return loss

def optimized_crnn_discriminator():
    """Optimized CRNN discriminator"""
    input_data = Input(name='input', shape=input_size_crnn, dtype='float32')
    
    # CNN layers
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_data)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Reshape untuk RNN
    new_shape = ((input_size_crnn[0] // 4), (input_size_crnn[1] // 4) * 128)
    reshape = Reshape(target_shape=new_shape)(pool2)
    
    # RNN layers
    dense1 = Dense(256, activation='relu')(reshape)
    
    # Output layer
    dense2 = Dense(len(charset_base) + 1, activation='softmax', name='dense2')(dense1)
    
    model = Model(inputs=input_data, outputs=dense2)
    return model

def get_optimized_gan_network(discriminator_1, discriminator_2, generator, optimizer):
    """Create optimized GAN network"""
    discriminator_1.trainable = False
    discriminator_2.trainable = False

    gan_input = Input(shape=(128, 1024, 1))

    out_generator = generator(gan_input)
    out_discrimintor_1 = discriminator_1([out_generator, gan_input])
    
    # Reshape untuk CRNN
    reshaped = Reshape((1024, 128, 1), input_shape=(128, 1024, 1))(out_generator)
    out_discrimintor_2 = discriminator_2(reshaped)

    gan = Model([gan_input], [out_discrimintor_1, out_generator, out_discrimintor_2])
    gan.compile(
        loss=['mse', 'binary_crossentropy', ctc_loss_lambda_func], 
        loss_weights=[1, 10, 1], 
        optimizer=optimizer
    )
    
    return gan

def optimized_train_gan(epochs=150, save_interval=10):
    """Optimized training function dengan multi-GPU"""
    print("=== OPTIMIZED GAN TRAINING ===")
    print(f"Global batch size: {GLOBAL_BATCH_SIZE}")
    print(f"Per-replica batch size: {PER_REPLICA_BATCH_SIZE}")
    print(f"CPU workers: {NUM_WORKERS}")
    print(f"GPU strategy: {strategy}")
    
    # Load data dengan parallel processing
    print("Loading training data...")
    list_image_train = glob(os.path.join('datasets/nan_distorted/train', '*.jpg'))
    list_lines = read_file('datasets/nan_raw_biner/train/lines.txt')
    
    print(f"Found {len(list_image_train)} training images")
    
    # Parallel image loading
    train_data = parallel_image_loader(list_image_train, split='train', num_workers=NUM_WORKERS)
    print(f"Successfully loaded {len(train_data)} image pairs")
    
    # Create optimized dataset
    train_dataset = create_optimized_dataset(train_data, list_lines, PER_REPLICA_BATCH_SIZE)
    
    # Distribute dataset
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    
    # Create models dalam strategy scope
    with strategy.scope():
        # Optimized learning rate untuk multi-GPU
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=2e-4 * strategy.num_replicas_in_sync,  # Scale dengan jumlah GPU
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        
        optimizer = Adam(learning_rate=lr_schedule, beta_1=0.5)
        
        print("Creating models...")
        generator = unet_generator()
        discriminator_1 = discriminator_patch()
        discriminator_2 = optimized_crnn_discriminator()
        
        # Compile discriminators
        discriminator_1.compile(
            loss='mse', 
            optimizer=Adam(learning_rate=lr_schedule), 
            metrics=['accuracy']
        )
        
        discriminator_2.compile(
            loss=ctc_loss_lambda_func,
            optimizer=Adam(learning_rate=lr_schedule)
        )
        
        gan = get_optimized_gan_network(discriminator_1, discriminator_2, generator, optimizer)
        
        print("Models created and compiled!")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            f"ResultGanS_{scenario}/epoch_{{epoch:03d}}/weights/generator.weights.h5",
            save_weights_only=True,
            save_freq='epoch',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(f"ResultGanS_{scenario}/training_log.csv")
    ]
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        start_time = time.time()
        
        # Training step
        epoch_losses = []
        batch_count = 0
        
        for batch_data in tqdm(train_dataset, desc=f"Epoch {epoch + 1}"):
            deg_images, gt_images, texts = batch_data
            
            # Train discriminator 1
            with strategy.scope():
                # Generate fake images
                generated_images = generator(deg_images, training=True)
                
                # Train discriminator 1
                real_loss = discriminator_1.train_on_batch([gt_images, deg_images], 
                                                         tf.ones_like(generated_images[:, :, :, :1]))
                fake_loss = discriminator_1.train_on_batch([generated_images, deg_images], 
                                                         tf.zeros_like(generated_images[:, :, :, :1]))
                d1_loss = 0.5 * (real_loss[0] + fake_loss[0])
                
                # Train discriminator 2 (CRNN)
                reshaped_gt = tf.reshape(gt_images, (-1, 1024, 128, 1))
                reshaped_gen = tf.reshape(generated_images, (-1, 1024, 128, 1))
                
                d2_real_loss = discriminator_2.train_on_batch(reshaped_gt, texts)
                d2_fake_loss = discriminator_2.train_on_batch(reshaped_gen, texts)
                d2_loss = 0.5 * (d2_real_loss + d2_fake_loss)
                
                # Train generator
                gan_loss = gan.train_on_batch(deg_images, [
                    tf.ones_like(generated_images[:, :, :, :1]),  # Adversarial loss
                    gt_images,  # Reconstruction loss
                    texts  # Text recognition loss
                ])
                
                epoch_losses.append({
                    'd1_loss': d1_loss,
                    'd2_loss': d2_loss,
                    'gan_loss': gan_loss[0]
                })
            
            batch_count += 1
            if batch_count >= 100:  # Limit untuk demo
                break
        
        # Print epoch statistics
        avg_d1_loss = np.mean([l['d1_loss'] for l in epoch_losses])
        avg_d2_loss = np.mean([l['d2_loss'] for l in epoch_losses])
        avg_gan_loss = np.mean([l['gan_loss'] for l in epoch_losses])
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"  D1 Loss: {avg_d1_loss:.4f}")
        print(f"  D2 Loss: {avg_d2_loss:.4f}")
        print(f"  GAN Loss: {avg_gan_loss:.4f}")
        
        # Save models setiap save_interval
        if (epoch + 1) % save_interval == 0:
            save_dir = f"ResultGanS_{scenario}/epoch_{epoch+1:03d}/weights"
            os.makedirs(save_dir, exist_ok=True)
            
            generator.save_weights(f"{save_dir}/generator.weights.h5")
            discriminator_1.save_weights(f"{save_dir}/discriminator_1.weights.h5")
            discriminator_2.save_weights(f"{save_dir}/discriminator_2.weights.h5")
            gan.save_weights(f"{save_dir}/gan.weights.h5")
            
            print(f"Models saved at epoch {epoch + 1}")
    
    print("\n=== Training completed! ===")
    
    # Final save
    final_save_dir = f"ResultGanS_{scenario}/final/weights"
    os.makedirs(final_save_dir, exist_ok=True)
    
    generator.save_weights(f"{final_save_dir}/generator.weights.h5")
    discriminator_1.save_weights(f"{final_save_dir}/discriminator_1.weights.h5")
    discriminator_2.save_weights(f"{final_save_dir}/discriminator_2.weights.h5")
    gan.save_weights(f"{final_save_dir}/gan.weights.h5")
    
    print("Final models saved!")
    
    return generator, discriminator_1, discriminator_2, gan

def main():
    """Main optimized training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Multi-GPU GAN-HTR Training')
    parser.add_argument('--epoch', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=GLOBAL_BATCH_SIZE, help='Global batch size')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval')
    
    args = parser.parse_args()
    
    print("=== OPTIMIZED MULTI-GPU GAN-HTR TRAINING ===")
    print(f"Total Epochs: {args.epoch}")
    print(f"Global Batch Size: {args.batch_size}")
    print(f"Hardware Resources:")
    print(f"  - CPU: AMD Threadripper PRO 3955WX (32 threads)")
    print(f"  - RAM: 128GB (104GB available)")
    print(f"  - GPU: 2x RTX A4000 (32GB total VRAM)")
    print(f"  - Workers: {NUM_WORKERS}")
    print(f"  - Strategy: {strategy}")
    
    # Start optimized training
    models = optimized_train_gan(epochs=args.epoch, save_interval=args.save_interval)
    
    print("\n🎉 OPTIMIZED TRAINING COMPLETED SUCCESSFULLY! 🎉")

if __name__ == "__main__":
    main()
