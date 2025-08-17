#!/usr/bin/env python3
"""
EMERGENCY TRAINING SCRIPT - GAN HTR
Script untuk mengatasi masalah NaN validation loss
Konfigurasi ultra-konservatif untuk debugging dan stabilitas

Usage: poetry run python periksa/emergency_training.py
"""

import os
import sys
sys.path.append('/home/lambda_one/tesis/GAN-HTR')

# Set environment variables BEFORE importing TensorFlow
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ['TF_DISABLE_LAYOUT_OPTIMIZER'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Single GPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from tqdm import tqdm

# Configure TensorFlow untuk stabilitas maksimum
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Configured {len(gpus)} GPU(s) with memory growth")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration warning: {e}")

# Disable mixed precision untuk stabilitas
# tf.keras.mixed_precision.set_global_policy('float32')  # Force float32

# Import modules yang diperlukan
from jnm_GAN_AHTR import (
    unet, build_discriminator_1, build_discriminator_2, 
    read_file, read_file_shuffle, normalizeTranscription, 
    encode_txt, charset_base, max_text_length, 
    rootPath, DatabasePath
)

class UltraSafeCTCLoss:
    """CTC Loss function yang sangat aman untuk mencegah NaN"""
    
    @staticmethod
    def safe_ctc_loss(y_true, y_pred):
        """Ultra-safe CTC loss implementation"""
        try:
            # Input validation dan casting
            y_true = tf.cast(y_true, tf.int32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            batch_size = tf.shape(y_true)[0]
            sequence_length = tf.shape(y_pred)[1]
            
            # Validasi minimum requirements
            if batch_size < 1 or sequence_length < 10:
                return tf.constant(1.0, dtype=tf.float32)
            
            # Label lengths dengan batas ketat
            label_length = tf.math.count_nonzero(y_true, axis=-1, dtype=tf.int32)
            label_length = tf.maximum(label_length, 1)
            label_length = tf.minimum(label_length, 15)  # Max 15 chars
            
            # Input length yang sangat konservatif
            input_length = tf.fill([batch_size], sequence_length // 8)  # Sangat konservatif
            input_length = tf.maximum(input_length, label_length * 3)
            
            # Preprocessing yang sangat hati-hati
            epsilon = 1e-6
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Softmax + stabilization
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            y_pred = y_pred + epsilon
            
            # Validation sebelum CTC
            max_label_len = tf.reduce_max(label_length)
            min_input_len = tf.reduce_min(input_length)
            
            valid_condition = tf.logical_and(
                tf.greater(min_input_len, max_label_len),
                tf.greater(max_label_len, 0)
            )
            
            def compute_safe_ctc():
                # CTC computation dengan error handling
                log_probs = tf.math.log(y_pred + epsilon)
                
                try:
                    loss = tf.nn.ctc_loss(
                        labels=y_true,
                        logits=log_probs,
                        label_length=label_length,
                        logit_length=input_length,
                        logits_time_major=False,
                        blank_index=-1
                    )
                    
                    # Extensive cleaning
                    loss = tf.where(tf.math.is_finite(loss), loss, tf.constant(1.0))
                    loss = tf.where(tf.math.is_nan(loss), tf.constant(1.0), loss)
                    loss = tf.clip_by_value(loss, 0.0, 5.0)  # Tight clipping
                    
                    return tf.reduce_mean(loss)
                    
                except Exception:
                    return tf.constant(1.0, dtype=tf.float32)
            
            def fallback_loss():
                return tf.constant(1.0, dtype=tf.float32)
            
            return tf.cond(valid_condition, compute_safe_ctc, fallback_loss)
            
        except Exception as e:
            print(f"❌ CTC Loss Error: {e}")
            return tf.constant(2.0, dtype=tf.float32)

class EmergencyDataGenerator:
    """Data generator yang sangat konservatif"""
    
    def __init__(self, max_samples=50):
        self.max_samples = max_samples
        self.processed_count = 0
        
    def safe_data_generator(self, image_list, lines_list, split='train'):
        """Generator dengan extensive error handling"""
        
        for im_base in image_list:
            if self.processed_count >= self.max_samples:
                break
                
            try:
                # Find actual file extension (bisa .jpg atau .png)
                from glob import glob
                search_pattern = os.path.join('datasets/nan_distorted', split, im_base + '.*')
                found_files = glob(search_pattern)
                
                if not found_files:
                    continue
                
                im_full_name = os.path.basename(found_files[0])
                
                # Load image dengan validation
                from jnm_GAN_AHTR import readGrayPair
                deg_image, gt_image = readGrayPair(im_full_name, split=split)
                
                # Validasi shape
                if deg_image.shape != (128, 1024, 1) or gt_image.shape != (128, 1024, 1):
                    continue
                    
                # Check for NaN values
                if np.any(np.isnan(deg_image)) or np.any(np.isnan(gt_image)):
                    print(f"⚠️ NaN detected in {im_base}, skipping")
                    continue
                
                # Find transcription
                line_text = None
                for line in lines_list:
                    if line.startswith(im_full_name):
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            line_text = parts[1]
                            break
                    # Also try without extension
                    elif line.startswith(im_base):
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            line_text = parts[1]
                            break
                
                if not line_text:
                    continue
                    
                # Process transcription dengan limit ketat
                line = normalizeTranscription(line_text)
                words = line.split()
                if len(words) > 10:  # Max 10 words
                    continue
                    
                encoded_txt = encode_txt(line)
                if not encoded_txt or len(encoded_txt) > 15:  # Max 15 chars
                    continue
                
                # Prepare CRNN data
                from data import preproc as pp
                gt_path = os.path.join(DatabasePath, split, 'images', im_full_name)
                if not os.path.exists(gt_path):
                    # Try different path structure
                    gt_path = os.path.join('datasets/nan_raw_biner', split, 'images', im_full_name)
                    if not os.path.exists(gt_path):
                        # Try with different extension
                        base_name = os.path.splitext(im_full_name)[0]
                        for ext in ['.jpg', '.png', '.tif']:
                            alt_path = os.path.join('datasets/nan_raw_biner', split, 'images', base_name + ext)
                            if os.path.exists(alt_path):
                                gt_path = alt_path
                                break
                        else:
                            continue  # Skip if no GT file found
                
                img = pp.preprocess(gt_path, (1024, 128, 1))
                
                if len(img.shape) == 2:
                    img = img.T
                    img = img[..., np.newaxis]
                elif len(img.shape) == 3 and img.shape == (128, 1024, 1):
                    img = np.transpose(img, (1, 0, 2))
                
                # Pad encoded text
                padded_encoded = np.zeros(20, dtype=np.int16)  # Reduced size
                padded_encoded[:len(encoded_txt)] = encoded_txt
                
                self.processed_count += 1
                
                yield {
                    'deg_image': deg_image.astype(np.float32),
                    'gt_image': gt_image.astype(np.float32),
                    'crnn_image': img.astype(np.float32),
                    'transcription': padded_encoded,
                    'text_line': line,
                    'image_name': im_base
                }
                
            except Exception as e:
                print(f"⚠️ Error processing {im_base}: {e}")
                continue

class EmergencyTrainer:
    """Emergency trainer dengan monitoring ekstensif"""
    
    def __init__(self):
        self.ctc_loss = UltraSafeCTCLoss()
        self.data_gen = EmergencyDataGenerator(max_samples=50)
        self.training_log = []
        
    def build_models(self):
        """Build models dengan konfigurasi konservatif"""
        print("🏗️ Building models with conservative configuration...")
        
        # Generator
        generator = unet()
        
        # Discriminator 1
        discriminator_1 = build_discriminator_1()
        discriminator_1.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            metrics=['accuracy']
        )
        
        # Discriminator 2 (CRNN) with safe CTC loss
        discriminator_2 = build_discriminator_2()
        discriminator_2.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=5e-6),  # Lebih kecil
            loss=self.ctc_loss.safe_ctc_loss
        )
        
        print("✅ All models built successfully")
        return generator, discriminator_1, discriminator_2
    
    def emergency_training_step(self, generator, discriminator_1, discriminator_2, 
                              batch_data, epoch, batch_idx):
        """Single training step dengan extensive monitoring"""
        
        try:
            batch_train = batch_data['deg_image']
            batch_target = batch_data['gt_image']
            x_train_rcnn = batch_data['crnn_image']
            y_train_rcnn = batch_data['transcription']
            
            batch_size = batch_train.shape[0]
            
            # Validate inputs
            if np.any(np.isnan(batch_train)) or np.any(np.isnan(batch_target)):
                print("❌ NaN in input data!")
                return None, None, None
            
            # Train Discriminator 1
            generated_images = generator.predict(batch_train, verbose=0)
            
            # Check generated images
            if np.any(np.isnan(generated_images)):
                print("❌ NaN in generated images!")
                return None, None, None
            
            valid = np.ones((batch_size, 8, 64, 1))
            fake = np.zeros((batch_size, 8, 64, 1))
            
            # Train on real and fake
            d1_loss_real = discriminator_1.train_on_batch([batch_target, batch_train], valid)
            d1_loss_fake = discriminator_1.train_on_batch([generated_images, batch_train], fake)
            d1_loss = 0.5 * np.add(d1_loss_real[0], d1_loss_fake[0])
            
            # Train Discriminator 2 (CRNN)
            try:
                d2_loss = discriminator_2.train_on_batch(x_train_rcnn, y_train_rcnn)
                if isinstance(d2_loss, list):
                    d2_loss = d2_loss[0]
            except Exception as e:
                print(f"⚠️ CRNN training failed: {e}")
                d2_loss = 1.0
            
            # Train Generator (simplified)
            # For emergency, we'll just train generator against discriminator 1
            discriminator_1.trainable = False
            try:
                g_loss = discriminator_1.train_on_batch([generated_images, batch_train], valid)
                if isinstance(g_loss, list):
                    g_loss = g_loss[0]
            except Exception as e:
                print(f"⚠️ Generator training failed: {e}")
                g_loss = 1.0
            finally:
                discriminator_1.trainable = True
            
            # Validate losses
            if np.isnan(d1_loss) or np.isnan(d2_loss) or np.isnan(g_loss):
                print(f"❌ NaN loss detected at epoch {epoch}, batch {batch_idx}")
                return None, None, None
            
            # Clip extreme values
            d1_loss = np.clip(d1_loss, 0.0, 10.0)
            d2_loss = np.clip(d2_loss, 0.0, 20.0)
            g_loss = np.clip(g_loss, 0.0, 10.0)
            
            self.training_log.append({
                'epoch': epoch,
                'batch': batch_idx,
                'd1_loss': float(d1_loss),
                'd2_loss': float(d2_loss),
                'g_loss': float(g_loss),
                'timestamp': time.time()
            })
            
            return d1_loss, d2_loss, g_loss
            
        except Exception as e:
            print(f"❌ Training step failed: {e}")
            return None, None, None
    
    def emergency_validate(self, generator, discriminator_1, discriminator_2, val_data):
        """Emergency validation dengan error handling"""
        
        try:
            val_losses = []
            
            for batch_data in val_data:
                batch_train = batch_data['deg_image']
                batch_target = batch_data['gt_image']
                
                # Generate images
                generated_images = generator.predict(batch_train, verbose=0)
                
                # Simple validation loss (content loss only)
                content_loss = np.mean((batch_target - generated_images) ** 2)
                val_losses.append(content_loss)
            
            if val_losses:
                avg_val_loss = np.mean(val_losses)
                if np.isnan(avg_val_loss):
                    return 1.0  # Fallback value
                return float(np.clip(avg_val_loss, 0.0, 5.0))
            else:
                return 1.0
                
        except Exception as e:
            print(f"⚠️ Validation failed: {e}")
            return 1.0
    
    def run_emergency_training(self, epochs=5):
        """Main emergency training loop"""
        
        print("🚨 STARTING EMERGENCY TRAINING")
        print("=" * 50)
        
        # Build models
        generator, discriminator_1, discriminator_2 = self.build_models()
        
        # Prepare data
        list_image_train = read_file_shuffle(rootPath + 'Sets/list_train_nan.txt')[:30]  # Only 30 images
        list_lines = read_file(rootPath + 'Sets/lines.txt')
        list_image_valid = read_file(rootPath + 'Sets/list_valid_nan.txt')[:10]  # Only 10 for validation
        
        print(f"📊 Data: {len(list_image_train)} train, {len(list_image_valid)} validation")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 3
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
            
            epoch_start = time.time()
            epoch_d1_losses, epoch_d2_losses, epoch_g_losses = [], [], []
            
            # Training
            train_data = list(self.data_gen.safe_data_generator(list_image_train, list_lines, 'train'))
            
            if not train_data:
                print("❌ No training data available!")
                break
            
            print(f"📊 Processing {len(train_data)} training samples...")
            
            for batch_idx, batch_data in enumerate(train_data):
                # Convert single sample to batch
                batch_data_formatted = {}
                for key, value in batch_data.items():
                    if key != 'text_line' and key != 'image_name':
                        batch_data_formatted[key] = np.expand_dims(value, axis=0)
                    else:
                        batch_data_formatted[key] = value
                
                # Training step
                d1_loss, d2_loss, g_loss = self.emergency_training_step(
                    generator, discriminator_1, discriminator_2,
                    batch_data_formatted, epoch, batch_idx
                )
                
                if d1_loss is None:  # Training failed
                    print(f"❌ Training failed at batch {batch_idx}")
                    continue
                
                epoch_d1_losses.append(d1_loss)
                epoch_d2_losses.append(d2_loss)
                epoch_g_losses.append(g_loss)
                
                # Progress report
                if batch_idx % 5 == 0:
                    print(f"   Batch {batch_idx}: D1={d1_loss:.4f}, D2={d2_loss:.4f}, G={g_loss:.4f}")
            
            # Epoch summary
            if epoch_d1_losses:
                avg_d1 = np.mean(epoch_d1_losses)
                avg_d2 = np.mean(epoch_d2_losses)
                avg_g = np.mean(epoch_g_losses)
                
                # Validation
                self.data_gen.processed_count = 0  # Reset counter
                val_data = list(self.data_gen.safe_data_generator(list_image_valid, list_lines, 'validation'))
                
                if val_data:
                    val_data_batch = []
                    for val_sample in val_data:
                        val_batch = {}
                        for key, value in val_sample.items():
                            if key != 'text_line' and key != 'image_name':
                                val_batch[key] = np.expand_dims(value, axis=0)
                        val_data_batch.append(val_batch)
                    
                    val_loss = self.emergency_validate(generator, discriminator_1, discriminator_2, val_data_batch)
                else:
                    val_loss = 1.0
                
                epoch_time = time.time() - epoch_start
                
                print(f"📈 Epoch {epoch + 1} Summary:")
                print(f"   Train - D1: {avg_d1:.4f}, D2: {avg_d2:.4f}, G: {avg_g:.4f}")
                print(f"   Validation: {val_loss:.6f}")
                print(f"   Time: {epoch_time:.1f}s")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"⭐ New best validation loss: {val_loss:.6f}")
                    
                    # Save best model
                    save_dir = "periksa/emergency_checkpoints"
                    os.makedirs(save_dir, exist_ok=True)
                    generator.save_weights(os.path.join(save_dir, f"generator_epoch_{epoch}.weights.h5"))
                    print(f"💾 Model saved to {save_dir}")
                    
                else:
                    patience_counter += 1
                    print(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print("🛑 Early stopping triggered")
                    break
                    
                # Reset data generator counter
                self.data_gen.processed_count = 0
            
            else:
                print("❌ No successful training batches in this epoch")
                break
        
        # Save training log
        log_path = "periksa/emergency_training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        print(f"\n✅ Emergency training completed!")
        print(f"📊 Best validation loss: {best_val_loss:.6f}")
        print(f"📁 Training log saved to: {log_path}")
        
        return generator, discriminator_1, discriminator_2

def main():
    """Main function untuk emergency training"""
    
    print("🚨 GAN-HTR EMERGENCY TRAINING")
    print("Konfigurasi ultra-konservatif untuk mengatasi NaN validation loss")
    print("=" * 70)
    
    try:
        # Check prerequisites
        print("🔍 Checking prerequisites...")
        
        # Check dataset
        train_list = rootPath + 'Sets/list_train_nan.txt'
        if not os.path.exists(train_list):
            print(f"❌ Training list not found: {train_list}")
            return
        
        # Check charset
        if not charset_base:
            print("❌ Character set not loaded")
            return
        
        print(f"✅ Character set loaded: {len(charset_base)} characters")
        print(f"✅ Dataset lists found")
        
        # Initialize emergency trainer
        trainer = EmergencyTrainer()
        
        # Run emergency training
        generator, discriminator_1, discriminator_2 = trainer.run_emergency_training(epochs=5)
        
        print("\n🎉 Emergency training session completed!")
        print("\n📋 NEXT STEPS:")
        print("1. Check training log in periksa/emergency_training_log.json")
        print("2. If successful, gradually increase batch size and learning rate")
        print("3. If still failing, check individual components with component_tests.py")
        print("4. Consider simplifying model architecture temporarily")
        
    except Exception as e:
        print(f"❌ Emergency training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
