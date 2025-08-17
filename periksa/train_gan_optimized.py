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
from tensorflow.keras.layers import LSTM, Bidirectional  # Explicit import for RNN layers
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
    """Fixed CTC loss using simple approach"""
    # Ensure y_true is 2D: (batch_size, max_label_length)
    if len(y_true.shape) > 2:
        y_true = tf.squeeze(y_true, axis=-1)
    
    # Get shapes
    batch_size = tf.shape(y_pred)[0]
    time_steps = tf.shape(y_pred)[1]  # Should be 128 from CRNN
    
    # Convert y_true to int32
    labels = tf.cast(y_true, tf.int32)
    
    # Create input_length: all samples have same time steps  
    input_length = tf.fill([batch_size], time_steps)
    
    # Calculate label_length: count non-zero elements per sample
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(labels, 0), tf.int32), axis=1)
    
    # Use simple CTC loss approach - back to K.ctc_batch_cost but with proper casting
    input_length = tf.cast(input_length, tf.float32)
    label_length = tf.cast(label_length, tf.float32)
    
    # Expand dimensions to match expected shape in Keras CTC
    input_length = tf.expand_dims(input_length, 1)  # (batch_size, 1)
    label_length = tf.expand_dims(label_length, 1)  # (batch_size, 1)
    
    loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    # Handle NaN values
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    
    return tf.reduce_mean(loss)

def optimized_crnn_discriminator():
    """Proper CRNN discriminator with temporal sequence output"""
    input_data = Input(name='input', shape=input_size_crnn, dtype='float32')
    
    # CNN feature extraction layers
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_data)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # (512, 64, 64)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # (256, 32, 128)
    
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)  # (128, 32, 256) - preserve width
    
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    
    # Reshape for RNN: (batch, width, height*channels) = (batch, 128, 32*512)
    # This gives us 128 time steps for CTC
    width = input_size_crnn[0] // 8  # 1024 // 8 = 128 time steps
    features = (input_size_crnn[1] // 4) * 512  # (128 // 4) * 512 = 32 * 512 = 16384
    reshape = Reshape(target_shape=(width, features))(conv4)
    
    # RNN layers for temporal modeling
    rnn1 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25))(reshape)
    rnn2 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25))(rnn1)
    
    # Dense layer for character prediction
    dense = Dense(len(charset_base) + 1, activation='softmax', name='dense2')(rnn2)
    
    model = Model(inputs=input_data, outputs=dense)
    return model

def get_optimized_gan_network(discriminator_1, discriminator_2, generator, optimizer):
    """Create optimized GAN network"""
    discriminator_1.trainable = False
    discriminator_2.trainable = False

    gan_input = Input(shape=(128, 1024, 1))

    out_generator = generator(gan_input)
    out_discrimintor_1 = discriminator_1([out_generator, gan_input])
    
    # Reshape untuk CRNN
    reshaped = Reshape((1024, 128, 1))(out_generator)
    out_discrimintor_2 = discriminator_2(reshaped)

    gan = Model([gan_input], [out_discrimintor_1, out_generator, out_discrimintor_2])
    gan.compile(
        loss=['mse', 'binary_crossentropy', ctc_loss_lambda_func], 
        loss_weights=[1, 10, 1], 
        optimizer=optimizer
    )
    
    return gan

def optimized_train_gan(epochs=150, save_interval=10, seed=42):
    """Optimized training function dengan multi-GPU"""
    # Set seeds for reproducibility
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
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
    
    # Create models & optimizers inside strategy scope
    with strategy.scope():
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=2e-4 * strategy.num_replicas_in_sync,
            decay_steps=2000,
            decay_rate=0.95,
            staircase=True
        )
        print("Creating models...")
        generator = unet_generator()
        discriminator_1 = discriminator_patch()
        discriminator_2 = optimized_crnn_discriminator()
        opt_g = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)
        opt_d1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)
        opt_d2 = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)
        mse = tf.keras.losses.MeanSquaredError()
        bce = tf.keras.losses.BinaryCrossentropy()
        print("Models created!")

        @tf.function
        def train_step(batch_data):
            deg_images, gt_images, texts = batch_data
            with tf.GradientTape(persistent=True) as tape:
                # Forward pass
                generated_images = generator(deg_images, training=True)
                real_pred = discriminator_1([gt_images, deg_images], training=True)
                fake_pred = discriminator_1([generated_images, deg_images], training=True)
                reshaped_gen = tf.reshape(generated_images, (-1, 1024, 128, 1))
                crnn_pred = discriminator_2(reshaped_gen, training=True)
                # Targets
                real_labels = tf.ones_like(real_pred)
                fake_labels = tf.zeros_like(fake_pred)
                # Discriminator 1 loss
                d1_real_loss = mse(real_labels, real_pred)
                d1_fake_loss = mse(fake_labels, fake_pred)
                d1_loss = 0.5 * (d1_real_loss + d1_fake_loss)
                # Generator losses
                adv_loss = mse(real_labels, fake_pred)
                recon_loss = bce(gt_images, generated_images)
                ctc = ctc_loss_lambda_func(texts, crnn_pred)
                g_total = adv_loss + 10.0 * recon_loss + ctc
            # Gradients
            grads_d1 = tape.gradient(d1_loss, discriminator_1.trainable_variables)
            grads_d2 = tape.gradient(ctc, discriminator_2.trainable_variables)
            grads_g = tape.gradient(g_total, generator.trainable_variables)
            # Apply
            opt_d1.apply_gradients(zip(grads_d1, discriminator_1.trainable_variables))
            opt_d2.apply_gradients(zip(grads_d2, discriminator_2.trainable_variables))
            opt_g.apply_gradients(zip(grads_g, generator.trainable_variables))
            return d1_loss, g_total, ctc, recon_loss, adv_loss

    # Metrics holders
    d1_metric = tf.keras.metrics.Mean(name='d1_loss')
    g_metric = tf.keras.metrics.Mean(name='g_loss')
    ctc_metric = tf.keras.metrics.Mean(name='ctc_loss')
    recon_metric = tf.keras.metrics.Mean(name='recon_loss')
    adv_metric = tf.keras.metrics.Mean(name='adv_loss')

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        start_time = time.time()
        # Reset metrics
        for m in [d1_metric, g_metric, ctc_metric, recon_metric, adv_metric]:
            m.reset_state()
        step = 0
        for batch_data in tqdm(train_dataset, desc=f"Epoch {epoch + 1}"):
            per_replica_losses = strategy.run(train_step, args=(batch_data,))
            # Reduce
            d1_l = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[0], axis=None)
            g_l = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[1], axis=None)
            ctc_l = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[2], axis=None)
            recon_l = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[3], axis=None)
            adv_l = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[4], axis=None)
            d1_metric.update_state(d1_l)
            g_metric.update_state(g_l)
            ctc_metric.update_state(ctc_l)
            recon_metric.update_state(recon_l)
            adv_metric.update_state(adv_l)
            step += 1
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} time: {epoch_time:.2f}s | D1: {d1_metric.result():.4f} | G: {g_metric.result():.4f} | Recon: {recon_metric.result():.4f} | Adv: {adv_metric.result():.4f} | CTC: {ctc_metric.result():.4f}")
        if (epoch + 1) % save_interval == 0:
            save_dir = f"ResultGanS_{scenario}/epoch_{epoch+1:03d}/weights"
            os.makedirs(save_dir, exist_ok=True)
            generator.save_weights(f"{save_dir}/generator.weights.h5")
            discriminator_1.save_weights(f"{save_dir}/discriminator_1.weights.h5")
            discriminator_2.save_weights(f"{save_dir}/discriminator_2.weights.h5")
            # Optionally export generator SavedModel
            generator.save(f"{save_dir}/generator_saved_model", save_format='tf')
            print(f"Models saved at epoch {epoch + 1}")
    
    print("\n=== Training completed! ===")
    
    # Final save
    final_save_dir = f"ResultGanS_{scenario}/final/weights"
    savedmodel_dir = f"ResultGanS_{scenario}/final/savedmodel"
    os.makedirs(final_save_dir, exist_ok=True)
    os.makedirs(savedmodel_dir, exist_ok=True)
    
    # Save weights
    generator.save_weights(f"{final_save_dir}/generator.weights.h5")
    discriminator_1.save_weights(f"{final_save_dir}/discriminator_1.weights.h5")
    discriminator_2.save_weights(f"{final_save_dir}/discriminator_2.weights.h5")
    
    # Save generator as SavedModel for deployment
    generator.save(f"{savedmodel_dir}/generator")
    
    print("Final models saved (weights + SavedModel)!")
    
    return generator, discriminator_1, discriminator_2

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
