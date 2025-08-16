#!/usr/bin/env python3
"""
Test script untuk memverifikasi perbaikan NaN losses
"""

import os
import sys
sys.path.append('/home/lambda_one/tesis/GAN-HTR')

def test_nan_fixes():
    """Test konfigurasi yang disarankan untuk menghindari NaN losses"""
    
    print("🧪 Testing GAN-HTR with NaN-resistant configuration...")
    
    # Configuration yang aman untuk mencegah NaN
    test_config = {
        'epochs': 5,           # Test singkat
        'batch_size': 8,       # Batch size yang stabil
        'learning_rate': 0.00005,  # LR konservatif
        'scenario': 'S_iam_OP_debug'
    }
    
    print(f"📋 Test Configuration:")
    for key, value in test_config.items():
        print(f"   {key}: {value}")
    
    # Build command
    cmd = f"""poetry run python jnm_GAN_AHTR.py \
        --epochs {test_config['epochs']} \
        --batch-size {test_config['batch_size']} \
        --learning-rate {test_config['learning_rate']} \
        --scenario {test_config['scenario']} \
        --patience 10 \
        --mode train"""
    
    print(f"\n🚀 Running command:")
    print(cmd)
    
    return cmd

def print_recommendations():
    """Print rekomendasi untuk mencegah NaN losses"""
    
    print("""
🔧 REKOMENDASI UNTUK MENCEGAH NaN LOSSES:

1. 📊 PARAMETER TRAINING:
   ✅ Learning Rate: 0.00005 - 0.0001 (konservatif)
   ✅ Batch Size: 8-16 (lebih besar = lebih stabil)
   ✅ Gradient Clipping: 0.5-1.0
   ✅ Loss Clipping: 0.0-10.0

2. 🎯 MONITORING YANG EFEKTIF:
   ✅ Report setiap 10 batch (bukan 25)
   ✅ Show learning rate dalam progress
   ✅ Display loss trends
   ✅ Emergency LR reduction jika NaN

3. 📈 TARGET LOSS RANGES:
   • D1 Loss: 0.3 - 2.0 (discriminator visual)
   • D2 Loss: 0.5 - 10.0 (CRNN recognition)
   • G Loss: 0.2 - 5.0 (generator)

4. 🚨 WARNING SIGNS:
   ⚠️  D1 < 0.1: Discriminator terlalu lemah
   ⚠️  D1 > 2.0: Discriminator kesulitan
   ⚠️  G > 5.0: Generator kesulitan
   ⚠️  D2 > 10.0: Recognition kesulitan

5. 🛠️ EMERGENCY ACTIONS:
   • Auto LR reduction 50% jika NaN
   • Gradient clipping yang ketat
   • Loss value clamping
   • Model weight validation

6. 💡 BEST PRACTICES:
   ✅ Mulai dengan konfigurasi konservatif
   ✅ Monitor setiap batch
   ✅ Use label smoothing (0.9/0.1)
   ✅ L1 loss lebih stabil dari MSE
   ✅ Validate model weights untuk NaN

COMMAND YANG DISARANKAN:
poetry run python jnm_GAN_AHTR.py --epochs 50 --batch-size 8 --learning-rate 0.00005 --scenario S_iam_OP_debug
    """)

if __name__ == "__main__":
    print_recommendations()
    cmd = test_nan_fixes()
    
    print(f"\n🎯 Untuk menjalankan test:")
    print(f"cd /home/lambda_one/tesis/GAN-HTR")
    print(f"{cmd}")
    
    print(f"\n📊 Monitor output untuk:")
    print(f"   • Tidak ada 'nan' dalam loss values")
    print(f"   • D1, D2, G dalam range yang wajar")
    print(f"   • Learning rate adjustments")
    print(f"   • Loss trends yang stabil")
