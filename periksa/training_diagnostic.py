import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os

class TrainingDiagnostic:
    """Diagnostic tool untuk menganalisis masalah training GAN-HTR"""
    
    def __init__(self, save_dir="periksa/diagnostic_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.loss_history = {
            'batch_times': [],
            'd1_losses': [],
            'd2_losses': [],
            'g_losses': [],
            'speeds': []
        }
    
    def diagnose_ctc_loss(self, y_true, y_pred, verbose=True):
        """Diagnose CTC loss issues"""
        try:
            # Basic shape analysis
            if verbose:
                print(f"🔍 CTC Loss Diagnosis:")
                print(f"   y_true shape: {y_true.shape}")
                print(f"   y_pred shape: {y_pred.shape}")
                
            # Check for valid labels
            y_true_int = tf.cast(y_true, tf.int32)
            label_lengths = tf.math.count_nonzero(y_true_int, axis=-1)
            valid_samples = tf.greater(label_lengths, 0)
            
            if verbose:
                print(f"   Valid samples: {tf.reduce_sum(tf.cast(valid_samples, tf.int32))}/{tf.shape(y_true)[0]}")
                print(f"   Label lengths: min={tf.reduce_min(label_lengths)}, max={tf.reduce_max(label_lengths)}")
                
            # Check prediction statistics
            pred_max = tf.reduce_max(y_pred)
            pred_min = tf.reduce_min(y_pred)
            pred_mean = tf.reduce_mean(y_pred)
            
            if verbose:
                print(f"   Predictions: min={pred_min:.4f}, max={pred_max:.4f}, mean={pred_mean:.4f}")
                
            # Check for problematic values
            has_inf = tf.reduce_any(tf.math.is_inf(y_pred))
            has_nan = tf.reduce_any(tf.math.is_nan(y_pred))
            
            if verbose:
                print(f"   Has inf: {has_inf}, Has nan: {has_nan}")
                
            return {
                'valid_samples': tf.reduce_sum(tf.cast(valid_samples, tf.int32)),
                'total_samples': tf.shape(y_true)[0],
                'min_pred': pred_min,
                'max_pred': pred_max,
                'has_inf': has_inf,
                'has_nan': has_nan
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ CTC diagnosis failed: {e}")
            return None
    
    def log_batch_performance(self, batch_idx, d1_loss, d2_loss, g_loss, batch_time, batch_size):
        """Log batch performance metrics"""
        speed = batch_size / batch_time if batch_time > 0 else 0
        
        self.loss_history['batch_times'].append(batch_time)
        self.loss_history['d1_losses'].append(float(d1_loss))
        self.loss_history['d2_losses'].append(float(d2_loss))
        self.loss_history['g_losses'].append(float(g_loss))
        self.loss_history['speeds'].append(speed)
        
        # Alert for problematic values
        if float(d2_loss) > 1000:
            print(f"🚨 CRITICAL: D2 loss explosion at batch {batch_idx}: {d2_loss:.2f}")
        
        if speed < 2.0:
            print(f"⚠️ SLOW: Very slow training at batch {batch_idx}: {speed:.1f} samples/sec")
    
    def suggest_fixes(self):
        """Suggest fixes based on observed patterns"""
        suggestions = []
        
        if len(self.loss_history['d2_losses']) > 10:
            recent_d2 = self.loss_history['d2_losses'][-10:]
            avg_d2 = np.mean(recent_d2)
            
            if avg_d2 > 500:
                suggestions.append("🔧 Reduce CRNN learning rate (try 0.0001)")
                suggestions.append("🔧 Increase gradient clipping for CRNN (try 0.1)")
                suggestions.append("🔧 Reduce CTC loss weight in GAN (try 1.0 instead of 10.0)")
            
            recent_speeds = self.loss_history['speeds'][-10:]
            avg_speed = np.mean(recent_speeds)
            
            if avg_speed < 5.0:
                suggestions.append("⚡ Reduce batch size for faster iteration")
                suggestions.append("⚡ Reduce dataset size for debugging")
                suggestions.append("⚡ Disable some optimizations temporarily")
        
        return suggestions
    
    def plot_training_progress(self):
        """Plot training progress"""
        if len(self.loss_history['d1_losses']) < 5:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Losses
        batches = range(len(self.loss_history['d1_losses']))
        ax1.plot(batches, self.loss_history['d1_losses'], label='D1 Loss', alpha=0.7)
        ax1.plot(batches, self.loss_history['g_losses'], label='G Loss', alpha=0.7)
        ax1.set_title('Discriminator 1 & Generator Losses')
        ax1.legend()
        ax1.set_yscale('log')
        
        # CRNN Loss (separate scale)
        ax2.plot(batches, self.loss_history['d2_losses'], label='D2 (CRNN) Loss', color='red', alpha=0.7)
        ax2.set_title('CRNN Loss')
        ax2.legend()
        ax2.set_yscale('log')
        
        # Speed
        ax3.plot(batches, self.loss_history['speeds'], label='Speed (samples/sec)', color='green', alpha=0.7)
        ax3.set_title('Training Speed')
        ax3.legend()
        
        # Batch times
        ax4.plot(batches, self.loss_history['batch_times'], label='Batch Time (s)', color='orange', alpha=0.7)
        ax4.set_title('Batch Processing Time')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_diagnostic.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Diagnostic plot saved to {self.save_dir}/training_diagnostic.png")

def create_emergency_training_config():
    """Create emergency configuration for unstable training"""
    return {
        'generator_lr': 0.00005,
        'discriminator_lr': 0.00005,
        'crnn_lr': 0.00001,
        'batch_size': 2,
        'gradient_clip_norm': 0.1,
        'loss_weights': [1.0, 1.0, 1.0],  # Reduced CRNN weight
        'use_mixed_precision': False  # Disable for stability
    }

def test_ctc_loss_stability():
    """Test CTC loss with sample data"""
    print("🧪 Testing CTC loss stability...")
    
    # Create sample data
    batch_size = 4
    seq_length = 256
    vocab_size = 80
    max_label_length = 20
    
    # Sample predictions (logits)
    y_pred = tf.random.normal((batch_size, seq_length, vocab_size))
    
    # Sample labels
    y_true = tf.random.uniform((batch_size, max_label_length), maxval=vocab_size-1, dtype=tf.int32)
    
    diagnostic = TrainingDiagnostic()
    result = diagnostic.diagnose_ctc_loss(y_true, y_pred)
    
    if result:
        print("✅ CTC loss test completed successfully")
        return True
    else:
        print("❌ CTC loss test failed")
        return False

if __name__ == "__main__":
    # Run basic tests
    test_ctc_loss_stability()
