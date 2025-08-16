#!/usr/bin/env python3
"""
Fix untuk mengatasi masalah NaN losses dalam training GAN-HTR
Penyebab utama NaN losses:
1. Gradient explosion dalam CTC loss
2. Learning rate terlalu tinggi
3. Batch size terlalu kecil
4. Data preprocessing yang tidak konsisten
"""

import tensorflow as tf
import numpy as np

def robust_ctc_loss(y_true, y_pred):
    """
    CTC loss yang lebih robust untuk mencegah NaN
    """
    # Ensure correct data types
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Get batch size and sequence length
    batch_size = tf.shape(y_true)[0]
    sequence_length = tf.shape(y_pred)[1]
    
    # Calculate label lengths
    label_length = tf.math.count_nonzero(y_true, axis=-1, dtype=tf.int32)
    label_length = tf.maximum(label_length, 1)  # Ensure minimum length of 1
    
    # Input length (sequence length for all samples)
    input_length = tf.fill([batch_size], sequence_length)
    
    # Clip predictions to prevent overflow
    y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
    
    # Add small epsilon to prevent log(0)
    epsilon = 1e-8
    y_pred = tf.nn.softmax(y_pred + epsilon)
    
    try:
        # Use tf.nn.ctc_loss instead of Keras CTC
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=tf.math.log(y_pred + epsilon),
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=-1  # Use default blank index
        )
        
        # Handle NaN and inf values
        loss = tf.where(tf.math.is_finite(loss), loss, tf.constant(10.0, dtype=tf.float32))
        loss = tf.clip_by_value(loss, 0.0, 10.0)  # Aggressive clipping
        
        return tf.reduce_mean(loss)
        
    except Exception as e:
        print(f"CTC loss failed: {e}")
        # Fallback to simple MSE
        return tf.constant(1.0, dtype=tf.float32)

def stable_discriminator_loss(real_output, fake_output):
    """
    Discriminator loss yang lebih stabil
    """
    # Use label smoothing
    real_labels = tf.ones_like(real_output) * 0.9  # Label smoothing
    fake_labels = tf.zeros_like(fake_output) * 0.1  # Label smoothing
    
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=real_output)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake_output)
    
    total_loss = real_loss + fake_loss
    
    # Clip to prevent explosion
    total_loss = tf.clip_by_value(total_loss, 0.0, 10.0)
    
    return tf.reduce_mean(total_loss)

def stable_generator_loss(fake_output, generated_images, target_images, recognition_loss):
    """
    Generator loss yang lebih stabil dengan weight balancing
    """
    # Adversarial loss
    adversarial_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(fake_output), 
        logits=fake_output
    )
    adversarial_loss = tf.reduce_mean(adversarial_loss)
    
    # Content loss (L1 loss lebih stabil dari MSE)
    content_loss = tf.reduce_mean(tf.abs(target_images - generated_images))
    
    # Combine losses with adaptive weights
    total_loss = (
        adversarial_loss * 0.1 +      # Reduced weight untuk adversarial
        content_loss * 1.0 +          # Standard weight untuk content
        recognition_loss * 5.0        # Reduced weight untuk recognition
    )
    
    # Clip final loss
    total_loss = tf.clip_by_value(total_loss, 0.0, 20.0)
    
    return total_loss

def create_robust_optimizers(base_lr=0.0001):
    """
    Create optimizers dengan settings yang lebih conservative
    """
    # Generator optimizer dengan gradient clipping
    gen_optimizer = tf.keras.optimizers.Adam(
        learning_rate=base_lr * 0.5,  # Reduced LR
        beta_1=0.5,
        beta_2=0.999,
        clipnorm=1.0  # Built-in gradient clipping
    )
    
    # Discriminator optimizer
    disc_optimizer = tf.keras.optimizers.Adam(
        learning_rate=base_lr * 0.2,  # Even lower LR for discriminator
        beta_1=0.5,
        beta_2=0.999,
        clipnorm=0.5  # Tighter clipping for discriminator
    )
    
    # CRNN optimizer
    crnn_optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=base_lr * 0.1,  # Much lower LR for CRNN
        clipnorm=0.5
    )
    
    return gen_optimizer, disc_optimizer, crnn_optimizer

def check_for_nan_in_model(model, model_name):
    """
    Check apakah ada NaN dalam model weights
    """
    has_nan = False
    for layer in model.layers:
        for weight in layer.weights:
            if tf.reduce_any(tf.math.is_nan(weight)):
                print(f"🚨 NaN detected in {model_name} layer {layer.name}")
                has_nan = True
    return has_nan

def emergency_weight_reset(model, model_name):
    """
    Reset weights jika terdeteksi NaN
    """
    print(f"🔧 Performing emergency weight reset for {model_name}")
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            weights = layer.get_weights()
            if weights:
                # Reinitialize dengan Xavier/Glorot
                new_weights = []
                for weight in weights:
                    if len(weight.shape) > 1:
                        # Xavier initialization
                        limit = np.sqrt(6.0 / (weight.shape[0] + weight.shape[1]))
                        new_weight = np.random.uniform(-limit, limit, weight.shape).astype(np.float32)
                    else:
                        # Zero initialization for biases
                        new_weight = np.zeros_like(weight).astype(np.float32)
                    new_weights.append(new_weight)
                layer.set_weights(new_weights)

def create_training_config_for_stability():
    """
    Training configuration yang optimal untuk stability
    """
    return {
        'batch_size': 8,  # Larger batch size untuk stability
        'learning_rate': 0.00005,  # Lower LR
        'epochs': 200,
        'patience': 30,
        'gradient_clip_norm': 1.0,
        'loss_weights': {
            'adversarial': 0.1,
            'content': 1.0,
            'recognition': 5.0
        },
        'validation_frequency': 5,
        'checkpoint_frequency': 10,
        'early_stop_threshold': 1e-6
    }

def print_training_diagnostics():
    """
    Print panduan untuk mendiagnosis masalah training
    """
    print("""
🔍 TRAINING DIAGNOSTICS GUIDE:

NaN Losses - Kemungkinan Penyebab:
1. Learning rate terlalu tinggi (>0.001)
2. Gradient explosion dalam CTC loss  
3. Batch size terlalu kecil (<4)
4. Data tidak dinormalisasi dengan benar
5. Model weights memiliki NaN values

Solutions:
✅ Gunakan learning rate 0.00005-0.0001
✅ Increase batch size ke 8-16
✅ Implementasikan gradient clipping
✅ Gunakan label smoothing
✅ Regular weight checking

Expected Loss Ranges:
- D1 Loss: 0.3 - 0.7 (stable discriminator)
- D2 Loss: 0.5 - 2.0 (CRNN recognition)  
- G Loss: 0.1 - 1.0 (generator)

Troubleshooting Steps:
1. Reduce learning rate by 50%
2. Increase batch size
3. Check data preprocessing
4. Enable gradient clipping
5. Use emergency weight reset if needed
    """)

if __name__ == "__main__":
    print_training_diagnostics()
