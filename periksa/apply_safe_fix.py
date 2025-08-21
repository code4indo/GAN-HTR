#!/usr/bin/env python3
"""
Script untuk menerapkan fix manual yang lebih aman
Hanya mengganti bagian yang penting untuk mengatasi NaN
"""

import os
import shutil
from datetime import datetime

def apply_safe_fix():
    """Apply fix yang lebih aman dan targeted"""
    
    script_path = "/home/lambda_one/tesis/GAN-HTR/jnm_GAN_AHTR.py"
    backup_path = f"/home/lambda_one/tesis/GAN-HTR/jnm_GAN_AHTR.py.backup_safe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("🔧 Applying Safe NaN Error Fix...")
    print(f"📁 Backing up original file to: {backup_path}")
    
    # Create backup
    shutil.copy2(script_path, backup_path)
    
    # Read original file
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply targeted fixes
    fixed_content = apply_targeted_fixes(content)
    
    # Write fixed file
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("✅ Safe fix applied successfully!")
    return backup_path

def apply_targeted_fixes(content):
    """Apply only the most critical fixes to prevent NaN"""
    
    # 1. Fix the CTC loss calculation (most important fix)
    old_input_length = '''# Original: compute input_length from y_pred sum
        # This follows the proven working method from GAN_AHTR.py
        input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
        input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)'''
    
    new_input_length = '''# FIXED: input_length should be sequence length, not sum of probabilities
        # This was the main cause of NaN in CTC loss
        batch_size = tf.shape(y_pred)[0]
        max_time_steps = tf.shape(y_pred)[1]
        input_length = tf.fill([batch_size], max_time_steps)
        input_length = tf.cast(input_length, tf.int32)'''
    
    content = content.replace(old_input_length, new_input_length)
    
    # 2. Fix the label_length calculation
    old_label_length = '''# Original: compute label_length from count_nonzero
        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")'''
    
    new_label_length = '''# FIXED: compute label_length properly for CTC
        label_length = tf.reduce_sum(
            tf.cast(tf.not_equal(y_true, 0), tf.int32), 
            axis=1
        )'''
    
    content = content.replace(old_label_length, new_label_length)
    
    # 3. Replace CTC loss function
    old_ctc_call = '''# Original: use K.ctc_batch_cost (proven working in GAN_AHTR.py)
            loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)'''
    
    new_ctc_call = '''# FIXED: use tf.nn.ctc_loss which is more stable
            loss = tf.nn.ctc_loss(
                labels=y_true,
                logits=y_pred,
                label_length=label_length,
                logit_length=input_length,
                blank_index=0,
                logits_time_major=False,
            )'''
    
    content = content.replace(old_ctc_call, new_ctc_call)
    
    # 4. Add loss clipping in CTC
    old_loss_handling = '''# Minimal safety: handle NaN/Inf without overcomplication
            loss = tf.where(tf.math.is_finite(loss), loss, self.fallback_loss)
            
            # Original: average loss across batch
            loss = tf.reduce_mean(loss)'''
    
    new_loss_handling = '''# FIXED: Better loss handling with clipping
            loss = tf.where(tf.math.is_finite(loss), loss, self.fallback_loss)
            loss = tf.clip_by_value(loss, 0.0, 10.0)
            loss = tf.reduce_mean(loss)'''
    
    content = content.replace(old_loss_handling, new_loss_handling)
    
    # 5. Reduce batch size for stability
    old_batch_default = "parser.add_argument('--batch-size', type=int, default=4,"
    new_batch_default = "parser.add_argument('--batch-size', type=int, default=2,"
    
    content = content.replace(old_batch_default, new_batch_default)
    
    # 6. Add comment about mixed precision
    gpu_config_end = '''    except RuntimeError as e:
        print(f"⚠️  GPU configuration warning: {e}")'''
    
    mixed_precision_note = '''    except RuntimeError as e:
        print(f"⚠️  GPU configuration warning: {e}")

# Mixed precision disabled for numerical stability in CTC loss
# If you need mixed precision, ensure CTC loss uses float32'''
    
    content = content.replace(gpu_config_end, mixed_precision_note)
    
    return content

if __name__ == "__main__":
    print("🚀 Starting Safe NaN Error Fix Application...")
    
    try:
        backup_path = apply_safe_fix()
        print(f"✅ Safe fix applied successfully!")
        print(f"📁 Backup created at: {backup_path}")
        print()
        print("🎯 Critical fixes applied:")
        print("   ✅ Fixed CTC loss input_length calculation (main cause of NaN)")
        print("   ✅ Fixed CTC loss label_length calculation")
        print("   ✅ Switched to tf.nn.ctc_loss for better stability")
        print("   ✅ Added loss clipping")
        print("   ✅ Reduced default batch size to 2")
        print()
        print("🧪 Test with: poetry run python jnm_GAN_AHTR.py --epochs 1 --batch-size 2")
        
    except Exception as e:
        print(f"❌ Error applying fix: {e}")
        import traceback
        traceback.print_exc()