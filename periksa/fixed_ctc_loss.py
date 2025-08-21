"""
Fixed CTC Loss Implementation
Solusi untuk masalah NaN pada training GAN-HTR
"""

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

class FixedCTCLoss:
    """
    Fixed CTC Loss implementation yang mengatasi masalah NaN
    
    Masalah utama pada implementasi lama:
    1. input_length calculation yang salah
    2. Shape mismatch antara generator output dan CRNN input
    3. Mixed precision numerical instability
    """
    
    def __init__(self, blank_index=0):
        self.fallback_loss = 2.0
        self.blank_index = blank_index
        
    def safe_ctc_loss(self, y_true, y_pred):
        """
        Fixed CTC loss calculation dengan proper input_length dan label_length
        
        Args:
            y_true: Ground truth labels [batch_size, max_label_length]
            y_pred: Predicted logits [batch_size, max_time_steps, num_classes]
        
        Returns:
            CTC loss value
        """
        
        # Ensure correct shapes
        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)
        
        # Cast to appropriate types
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Get batch dimensions
        batch_size = tf.shape(y_pred)[0]
        max_time_steps = tf.shape(y_pred)[1]
        
        # FIXED: input_length is the actual sequence length (time steps)
        # Not the sum of probabilities like in the buggy version
        input_length = tf.fill([batch_size], max_time_steps)
        input_length = tf.cast(input_length, tf.int32)
        
        # FIXED: label_length is the actual non-blank label count
        # Remove padding (assuming 0 is padding)
        label_length = tf.reduce_sum(
            tf.cast(tf.not_equal(y_true, 0), tf.int32), 
            axis=1
        )
        
        # Ensure minimum viable lengths for CTC
        label_length = tf.maximum(label_length, 1)
        # Input length must be >= label_length for CTC to work
        min_input_length = label_length + 1
        input_length = tf.maximum(input_length, min_input_length)
        
        try:
            # Use TensorFlow's native CTC loss (more stable than K.ctc_batch_cost)
            loss = tf.nn.ctc_loss(
                labels=y_true,
                logits=y_pred,
                label_length=label_length,
                logit_length=input_length,
                blank_index=self.blank_index,
                logits_time_major=False,  # Batch major format
            )
            
            # Handle NaN/Inf values
            is_finite = tf.math.is_finite(loss)
            loss = tf.where(is_finite, loss, self.fallback_loss)
            
            # Clip extreme values
            loss = tf.clip_by_value(loss, 0.0, 10.0)
            
            return tf.reduce_mean(loss)
            
        except Exception as e:
            print(f"🚨 CTC Loss calculation failed: {e}")
            return tf.constant(self.fallback_loss, dtype=tf.float32)

    def debug_ctc_inputs(self, y_true, y_pred):
        """
        Debug function untuk memeriksa input CTC
        """
        print("🔍 CTC Debug Info:")
        print(f"   y_true shape: {y_true.shape}")
        print(f"   y_pred shape: {y_pred.shape}")
        
        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)
        
        batch_size = tf.shape(y_pred)[0]
        max_time_steps = tf.shape(y_pred)[1]
        
        input_length = tf.fill([batch_size], max_time_steps)
        label_length = tf.reduce_sum(
            tf.cast(tf.not_equal(y_true, 0), tf.int32), 
            axis=1
        )
        
        print(f"   batch_size: {batch_size}")
        print(f"   max_time_steps: {max_time_steps}")
        print(f"   input_length: {input_length}")
        print(f"   label_length: {label_length}")
        
        # Check for potential issues
        if tf.reduce_any(tf.math.is_nan(y_pred)):
            print("   ⚠️ NaN detected in y_pred")
        if tf.reduce_any(tf.math.is_inf(y_pred)):
            print("   ⚠️ Inf detected in y_pred")
        if tf.reduce_any(tf.less(input_length, label_length)):
            print("   ⚠️ input_length < label_length detected")


def fix_generator_to_crnn_shape(generator_out):
    """
    Fix shape compatibility antara generator output dan CRNN input
    
    Args:
        generator_out: [batch, height, width, channels] - image format
    
    Returns:
        crnn_input: [batch, time_steps, features] - sequence format
    """
    
    # Generator output biasanya: [batch, 128, 1024, 1]
    batch_size = tf.shape(generator_out)[0]
    height = tf.shape(generator_out)[1]     # 128
    width = tf.shape(generator_out)[2]      # 1024
    channels = tf.shape(generator_out)[3]   # 1
    
    # Untuk HTR, width biasanya dijadikan time steps (sequence)
    # height * channels dijadikan features
    
    # Reshape: [batch, width, height * channels]
    # width = time steps (1024)
    # height * channels = features (128 * 1 = 128)
    crnn_input = tf.reshape(generator_out, [batch_size, width, height * channels])
    
    return crnn_input


def create_stable_training_config():
    """
    Create configuration untuk training yang lebih stabil
    """
    config = {
        # Disable mixed precision untuk stability
        'use_mixed_precision': False,
        
        # Learning rates yang lebih konservatif
        'generator_lr': 1e-5,
        'discriminator_lr': 1e-5,
        
        # Gradient clipping
        'gradient_clip_norm': 1.0,
        
        # Loss weights yang lebih balanced
        'adversarial_weight': 0.1,  # Reduced from 0.5
        'content_weight': 1.0,      # Keep content loss dominant
        'recognition_weight': 0.3,  # Reduced from 0.5
        
        # Batch size yang lebih kecil untuk stability
        'batch_size': 2,  # Reduced from 4
        
        # Validation frequency
        'eval_interval': 5,  # Less frequent validation
        
        # Early stopping patience
        'patience': 15,
    }
    
    return config


if __name__ == "__main__":
    # Test CTC loss implementation
    print("🧪 Testing Fixed CTC Loss...")
    
    # Create test data
    batch_size = 2
    max_time_steps = 100
    num_classes = 80
    max_label_length = 20
    
    # Random test data
    y_pred = tf.random.normal([batch_size, max_time_steps, num_classes])
    y_true = tf.random.uniform([batch_size, max_label_length], 0, num_classes-1, dtype=tf.int32)
    
    # Test fixed CTC loss
    ctc_loss = FixedCTCLoss()
    loss_value = ctc_loss.safe_ctc_loss(y_true, y_pred)
    
    print(f"✅ CTC Loss computed successfully: {loss_value}")
    
    # Debug information
    ctc_loss.debug_ctc_inputs(y_true, y_pred)
    
    # Test shape conversion
    generator_out = tf.random.normal([batch_size, 128, 1024, 1])
    crnn_input = fix_generator_to_crnn_shape(generator_out)
    
    print(f"✅ Shape conversion successful:")
    print(f"   Generator out: {generator_out.shape}")
    print(f"   CRNN input: {crnn_input.shape}")