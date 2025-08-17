#!/usr/bin/env python3
"""
Fix untuk masalah FAILED_PRECONDITION pada optimizer variables dalam distributed training.

Error yang terjadi:
- Read variable failure adam/batch_normalization_2_gamma_momentum/replica_1/576
- Variable is uninitialized or on another device

Penyebab:
1. Optimizer variables tidak ter-inisialisasi dengan benar dalam distribusi strategy
2. Model tidak di-build secara eksplisit sebelum training dimulai
3. Optimizer state tidak sinkron antar replicas

Solusi:
1. Eksplisit build model dengan data sample
2. Inisialisasi optimizer variables
3. Dummy forward pass untuk warmup
"""

import tensorflow as tf
import numpy as np

def fix_optimizer_initialization_issue():
    """
    Analisis masalah dan solusi untuk optimizer variable initialization
    """
    
    print("🔍 ANALISIS MASALAH OPTIMIZER VARIABLES")
    print("=" * 60)
    
    print("\n❌ MASALAH YANG TERJADI:")
    print("1. adam/batch_normalization_2_gamma_momentum/replica_1/576 tidak ter-inisialisasi")
    print("2. Variable berada di device yang berbeda atau belum diinisialisasi") 
    print("3. tf2xla conversion gagal saat kompilasi XLA")
    print("4. Distributed training dengan MirroredStrategy mengalami masalah sinkronisasi")
    
    print("\n🔍 PENYEBAB ROOT CAUSE:")
    print("1. Model tidak di-build secara eksplisit sebelum training")
    print("2. Optimizer variables tidak ter-inisialisasi dengan benar")
    print("3. Batch normalization layers membutuhan warmup untuk momentum variables")
    print("4. Strategy.scope() tidak mencakup semua inisialisasi yang diperlukan")
    
    print("\n✅ SOLUSI YANG DIPERLUKAN:")
    print("1. Eksplisit build model dengan sample data")
    print("2. Dummy forward pass untuk inisialisasi BN layers")
    print("3. Inisialisasi optimizer variables secara eksplisit")
    print("4. Sinkronisasi variables antar replicas")
    
    return True

def create_model_builder_fix():
    """
    Membuat fungsi untuk fix model building dan optimizer initialization
    """
    
    fix_code = """
def build_and_initialize_models_properly(strategy, input_shape=(64, 512, 1)):
    '''
    Properly build and initialize all models with their optimizer variables
    '''
    print("🔧 Building and initializing models with proper variable initialization...")
    
    with strategy.scope():
        # Create models
        generator = unet()
        discriminator_1 = build_discriminator_1() 
        discriminator_2 = build_discriminator_2()
        
        # Create sample data for building models
        sample_input = tf.random.normal((1,) + input_shape)
        sample_target = tf.random.normal((1,) + input_shape)
        sample_crnn_input = tf.random.normal((1, 64, 512, 1))
        
        print("🏗️ Building generator with sample data...")
        _ = generator(sample_input, training=False)
        
        print("🏗️ Building discriminator_1 with sample data...")
        _ = discriminator_1([sample_target, sample_input], training=False)
        
        print("🏗️ Building discriminator_2 with sample data...")  
        _ = discriminator_2(sample_crnn_input, training=False)
        
        # Create optimizers AFTER models are built
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
        disc1_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
        disc2_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
        
        # Initialize optimizer variables with dummy gradients
        print("🔧 Initializing optimizer variables...")
        
        # Generator optimizer initialization
        with tf.GradientTape() as tape:
            fake_output = generator(sample_input, training=True)
            fake_loss = tf.reduce_mean(tf.square(fake_output - sample_target))
        
        gen_grads = tape.gradient(fake_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
        
        # Discriminator 1 optimizer initialization  
        with tf.GradientTape() as tape:
            real_pred = discriminator_1([sample_target, sample_input], training=True)
            fake_pred = discriminator_1([fake_output, sample_input], training=True)
            d1_loss = tf.reduce_mean(tf.square(real_pred - 1.0)) + tf.reduce_mean(tf.square(fake_pred))
            
        d1_grads = tape.gradient(d1_loss, discriminator_1.trainable_variables)
        disc1_optimizer.apply_gradients(zip(d1_grads, discriminator_1.trainable_variables))
        
        # Discriminator 2 optimizer initialization
        with tf.GradientTape() as tape:
            crnn_pred = discriminator_2(sample_crnn_input, training=True)
            d2_loss = tf.reduce_mean(tf.square(crnn_pred))
            
        d2_grads = tape.gradient(d2_loss, discriminator_2.trainable_variables)
        disc2_optimizer.apply_gradients(zip(d2_grads, discriminator_2.trainable_variables))
        
        print("✅ All models and optimizers properly initialized!")
        
        return generator, discriminator_1, discriminator_2, gen_optimizer, disc1_optimizer, disc2_optimizer
"""
    
    return fix_code

def create_training_step_fix():
    """
    Membuat training step yang lebih robust
    """
    
    fix_code = """
@tf.function  
def robust_distributed_train_step(batch_data, gen_optimizer, disc1_optimizer, disc2_optimizer):
    '''
    Robust distributed training step with proper variable handling
    '''
    
    def train_step(inputs):
        batch_train = inputs['deg_image']
        batch_target = inputs['gt_image'] 
        x_train_rcnn = inputs['crnn_image']
        y_train_rcnn = inputs['transcription']
        
        per_replica_batch_size = tf.shape(batch_train)[0]
        
        # Ensure all variables are on the same device
        with tf.device('/gpu:0'):  # Force specific device if needed
            
            # Train discriminator_1
            with tf.GradientTape() as disc1_tape:
                # Generate images first
                generated_images = generator(batch_train, training=False)
                
                # Get predictions
                real_pred = discriminator_1([batch_target, batch_train], training=True)
                fake_pred = discriminator_1([generated_images, batch_train], training=True)
                
                # Calculate discriminator loss
                valid = tf.ones_like(real_pred)
                fake = tf.zeros_like(fake_pred)
                
                d1_real_loss = tf.keras.losses.binary_crossentropy(valid, real_pred)
                d1_fake_loss = tf.keras.losses.binary_crossentropy(fake, fake_pred)
                d1_loss = tf.reduce_mean(d1_real_loss + d1_fake_loss)
                d1_loss = tf.clip_by_value(d1_loss, 0.0, 10.0)
            
            # Apply discriminator 1 gradients
            disc1_grads = disc1_tape.gradient(d1_loss, discriminator_1.trainable_variables)
            disc1_grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in disc1_grads]
            
            # Filter out None gradients
            disc1_grads_and_vars = [(grad, var) for grad, var in zip(disc1_grads, discriminator_1.trainable_variables) if grad is not None]
            if disc1_grads_and_vars:
                disc1_optimizer.apply_gradients(disc1_grads_and_vars)
            
            # Train discriminator_2 (CRNN)
            with tf.GradientTape() as disc2_tape:
                crnn_pred = discriminator_2(x_train_rcnn, training=True)
                d2_loss = tf.keras.losses.CTC(y_train_rcnn, crnn_pred, logits_time_major=False)
                d2_loss = tf.reduce_mean(d2_loss)
                d2_loss = tf.clip_by_value(d2_loss, 0.0, 100.0)
            
            # Apply discriminator 2 gradients  
            disc2_grads = disc2_tape.gradient(d2_loss, discriminator_2.trainable_variables)
            disc2_grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in disc2_grads]
            
            disc2_grads_and_vars = [(grad, var) for grad, var in zip(disc2_grads, discriminator_2.trainable_variables) if grad is not None]
            if disc2_grads_and_vars:
                disc2_optimizer.apply_gradients(disc2_grads_and_vars)
            
            # Train generator
            with tf.GradientTape() as gen_tape:
                generated_images = generator(batch_train, training=True)
                content_loss = tf.reduce_mean(tf.abs(batch_target - generated_images))
                g_loss = content_loss * 1.0
                g_loss = tf.clip_by_value(g_loss, 0.0, 100.0)
            
            # Apply generator gradients
            gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
            gen_grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in gen_grads]
            
            gen_grads_and_vars = [(grad, var) for grad, var in zip(gen_grads, generator.trainable_variables) if grad is not None]
            if gen_grads_and_vars:
                gen_optimizer.apply_gradients(gen_grads_and_vars)
        
        return d1_loss, d2_loss, g_loss
    
    # Run distributed training step
    per_replica_losses = strategy.run(train_step, args=(batch_data,))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
"""
    
    return fix_code

def main():
    """
    Main function untuk analisis dan solusi
    """
    
    print("🚀 ANALISIS DAN SOLUSI MASALAH OPTIMIZER VARIABLES")
    print("=" * 70)
    
    # Analisis masalah
    fix_optimizer_initialization_issue()
    
    print("\n📝 KODE PERBAIKAN YANG DIPERLUKAN:")
    print("=" * 50)
    
    # Model builder fix
    print("\n1. MODEL BUILDER FIX:")
    print(create_model_builder_fix())
    
    # Training step fix  
    print("\n2. TRAINING STEP FIX:")
    print(create_training_step_fix())
    
    print("\n🎯 LANGKAH IMPLEMENTASI:")
    print("=" * 30)
    print("1. Ganti fungsi create models dengan build_and_initialize_models_properly()")
    print("2. Ganti distributed_train_step dengan robust_distributed_train_step()")
    print("3. Pastikan semua model di-build dengan sample data sebelum training")
    print("4. Inisialisasi optimizer variables dengan dummy gradients")
    print("5. Gunakan device placement yang eksplisit jika diperlukan")
    
    print("\n✅ HASIL YANG DIHARAPKAN:")
    print("- Optimizer variables ter-inisialisasi dengan benar")
    print("- Batch normalization momentum variables tersedia di semua replicas") 
    print("- Tidak ada lagi FAILED_PRECONDITION error")
    print("- Training berjalan stabil dengan distributed strategy")

if __name__ == "__main__":
    main()
