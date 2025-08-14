"""
Implementasi Multi-GPU Training untuk GAN-HTR
==============================================

Modifikasi untuk menggunakan kedua GPU RTX A4000 secara bersamaan
"""

import tensorflow as tf
import numpy as np
import gc
import os

def configure_multi_gpu_setup():
    """
    Konfigurasi optimal untuk dual RTX A4000
    """
    
    # Set environment variables untuk optimasi
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Deteksi GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Detected GPUs: {len(gpus)}")
    
    if gpus:
        try:
            # Konfigurasi memory growth untuk semua GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Opsional: Set memory limit jika diperlukan
            # for gpu in gpus:
            #     tf.config.experimental.set_memory_limit(gpu, 14000)  # 14GB limit
            
            print("GPU memory growth configured successfully")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Setup distributed strategy
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
        
        # Enable mixed precision untuk semua devices
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled across all devices")
        
        return strategy
    else:
        print("Warning: Only single GPU detected, using OneDeviceStrategy")
        return tf.distribute.OneDeviceStrategy("/gpu:0")

def create_optimized_dataset_for_multi_gpu(data_generator, global_batch_size, strategy):
    """
    Membuat dataset yang dioptimasi untuk multi-GPU training
    
    Args:
        data_generator: Generator function untuk data
        global_batch_size: Total batch size (akan dibagi across GPUs)
        strategy: TensorFlow distribute strategy
    """
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Batch size per replica
    batch_size_per_replica = global_batch_size // strategy.num_replicas_in_sync
    print(f"Global batch size: {global_batch_size}")
    print(f"Batch size per replica: {batch_size_per_replica}")
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature={
            'deg_image': tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
            'gt_image': tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
            'crnn_image': tf.TensorSpec(shape=(1024, 128, 1), dtype=tf.float32),
            'transcription': tf.TensorSpec(shape=(None,), dtype=tf.int16),
            'text_line': tf.TensorSpec(shape=(), dtype=tf.string)
        }
    )
    
    # Pipeline optimizations
    dataset = dataset.cache()  # Cache in memory untuk performa
    dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size_per_replica, drop_remainder=True)
    dataset = dataset.repeat()  # Infinite dataset untuk training continuous
    dataset = dataset.prefetch(AUTOTUNE)
    
    # Distribute dataset
    distributed_dataset = strategy.experimental_distribute_dataset(dataset)
    
    return distributed_dataset

class MultiGPUGANTrainer:
    """
    Trainer class yang dioptimasi untuk multi-GPU GAN training
    """
    
    def __init__(self, strategy):
        self.strategy = strategy
        self.global_batch_size = 8  # 4 per GPU untuk dual-GPU
        
    def create_models_in_strategy_scope(self, generator_fn, discriminator1_fn, discriminator2_fn, gan_fn):
        """
        Membuat model dalam strategy scope untuk distributed training
        """
        with self.strategy.scope():
            # Create models
            generator = generator_fn()
            discriminator_1 = discriminator1_fn()
            discriminator_2 = discriminator2_fn()
            gan = gan_fn(generator, discriminator_1, discriminator_2)
            
            # Create optimizers
            gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
            disc1_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
            disc2_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
            
            return {
                'generator': generator,
                'discriminator_1': discriminator_1,
                'discriminator_2': discriminator_2,
                'gan': gan,
                'gen_optimizer': gen_optimizer,
                'disc1_optimizer': disc1_optimizer,
                'disc2_optimizer': disc2_optimizer
            }
    
    @tf.function
    def distributed_train_step(self, models, batch_data):
        """
        Training step yang didistribusi across multiple GPUs
        """
        def train_step(inputs):
            batch_train = inputs['deg_image']
            batch_target = inputs['gt_image']
            x_train_rcnn = inputs['crnn_image']
            y_train_rcnn = inputs['transcription']
            
            batch_size = tf.shape(batch_train)[0]
            
            # Generate images
            generated_images = models['generator'](batch_train, training=False)
            
            # Prepare labels
            valid = tf.ones((batch_size, 8, 64, 1))
            fake = tf.zeros((batch_size, 8, 64, 1))
            
            # Train discriminator_1
            with tf.GradientTape() as disc1_tape:
                real_pred = models['discriminator_1']([batch_target, batch_train], training=True)
                fake_pred = models['discriminator_1']([generated_images, batch_train], training=True)
                
                d1_loss_real = tf.keras.losses.binary_crossentropy(valid, real_pred)
                d1_loss_fake = tf.keras.losses.binary_crossentropy(fake, fake_pred)
                d1_loss = (d1_loss_real + d1_loss_fake) / 2
                d1_loss = tf.reduce_mean(d1_loss)
            
            # Apply discriminator_1 gradients
            disc1_grads = disc1_tape.gradient(d1_loss, models['discriminator_1'].trainable_variables)
            models['disc1_optimizer'].apply_gradients(zip(disc1_grads, models['discriminator_1'].trainable_variables))
            
            # Train discriminator_2 (CRNN)
            with tf.GradientTape() as disc2_tape:
                d2_loss = models['discriminator_2'](x_train_rcnn, y_train_rcnn, training=True)
                if isinstance(d2_loss, list):
                    d2_loss = d2_loss[0]
                d2_loss = tf.reduce_mean(d2_loss)
            
            # Apply discriminator_2 gradients
            disc2_grads = disc2_tape.gradient(d2_loss, models['discriminator_2'].trainable_variables)
            models['disc2_optimizer'].apply_gradients(zip(disc2_grads, models['discriminator_2'].trainable_variables))
            
            # Train generator
            with tf.GradientTape() as gen_tape:
                g_loss = models['gan']([batch_train], [valid, batch_target, y_train_rcnn], training=True)
                if isinstance(g_loss, list):
                    g_loss = g_loss[0]
                g_loss = tf.reduce_mean(g_loss)
            
            # Apply generator gradients
            gen_grads = gen_tape.gradient(g_loss, models['generator'].trainable_variables)
            models['gen_optimizer'].apply_gradients(zip(gen_grads, models['generator'].trainable_variables))
            
            return d1_loss, d2_loss, g_loss
        
        # Run distributed training step
        per_replica_losses = self.strategy.run(train_step, args=(batch_data,))
        
        # Reduce losses across replicas
        d1_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[0], axis=None)
        d2_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[1], axis=None)
        g_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[2], axis=None)
        
        return d1_loss, d2_loss, g_loss

def optimize_tf_data_pipeline():
    """
    Optimasi TensorFlow data pipeline untuk performa maksimal
    """
    # Set thread pool sizes untuk CPU utilization optimal
    tf.config.threading.set_intra_op_parallelism_threads(16)  # Setengah dari total threads
    tf.config.threading.set_inter_op_parallelism_threads(16)  # Setengah dari total threads
    
    print("TensorFlow threading optimized for Threadripper PRO")

def memory_optimization_settings():
    """
    Pengaturan optimasi memori untuk training yang stabil
    """
    # Enable memory optimization
    tf.config.optimizer.set_jit(True)  # XLA JIT compilation
    tf.config.experimental.enable_tensor_float_32()  # TF32 untuk RTX series
    
    print("Memory and compute optimizations enabled")

# Template untuk mengintegrasikan ke script utama
INTEGRATION_TEMPLATE = '''
# Tambahkan di awal script utama jnm_GAN_AHTR.py

# Import fungsi optimasi
from periksa.multi_gpu_optimization import (
    configure_multi_gpu_setup,
    create_optimized_dataset_for_multi_gpu,
    MultiGPUGANTrainer,
    optimize_tf_data_pipeline,
    memory_optimization_settings
)

# Setup optimasi di awal main()
def main():
    # Optimasi TensorFlow
    optimize_tf_data_pipeline()
    memory_optimization_settings()
    
    # Setup multi-GPU
    strategy = configure_multi_gpu_setup()
    
    # Create trainer
    trainer = MultiGPUGANTrainer(strategy)
    
    # Create models dalam strategy scope
    models = trainer.create_models_in_strategy_scope(
        create_generator,
        create_discriminator_1, 
        create_discriminator_2,
        create_gan
    )
    
    # Create optimized dataset
    dataset = create_optimized_dataset_for_multi_gpu(
        data_generator=your_data_generator,
        global_batch_size=8,  # 4 per GPU
        strategy=strategy
    )
    
    # Training loop
    for epoch in range(epochs):
        for batch_data in dataset:
            d1_loss, d2_loss, g_loss = trainer.distributed_train_step(models, batch_data)
            
            # Print losses
            if step % 10 == 0:
                print(f"D1: {d1_loss:.4f}, D2: {d2_loss:.4f}, G: {g_loss:.4f}")

if __name__ == "__main__":
    main()
'''

if __name__ == "__main__":
    print("Multi-GPU Optimization Module for GAN-HTR")
    print("="*50)
    print("Functions available:")
    print("- configure_multi_gpu_setup()")
    print("- create_optimized_dataset_for_multi_gpu()")
    print("- MultiGPUGANTrainer class")
    print("- optimize_tf_data_pipeline()")
    print("- memory_optimization_settings()")
    print("\nIntegration template available in INTEGRATION_TEMPLATE variable")
