#!/usr/bin/env python3
"""
Test script untuk memverifikasi perbaikan loss clipping
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_training_logs():
    """Analyze training logs untuk melihat apakah loss masih stuck di clipping limits"""
    
    print("🔍 ANALISIS MASALAH LOSS CLIPPING")
    print("="*50)
    
    # Baca training log terbaru
    try:
        with open('/home/lambda_one/tesis/GAN-HTR/training_logs/training_log_1755359367.json', 'r') as f:
            data = json.load(f)
            
        epochs = data['epoch']
        d1_losses = data['d1_loss']
        d2_losses = data['d2_loss']
        g_losses = data['g_loss']
        
        print(f"📊 Data dari {len(epochs)} epochs:")
        print(f"   D1 Loss range: {min(d1_losses):.4f} - {max(d1_losses):.4f}")
        print(f"   D2 Loss range: {min(d2_losses):.4f} - {max(d2_losses):.4f}")
        print(f"   G Loss range: {min(g_losses):.4f} - {max(g_losses):.4f}")
        
        # Check untuk stuck values
        print(f"\n🚨 DIAGNOSIS MASALAH:")
        
        # D2 Loss stuck check
        d2_stuck = all(loss == 10.0 for loss in d2_losses)
        if d2_stuck:
            print(f"   ❌ D2 Loss STUCK di 10.0 (clipping limit)")
        else:
            print(f"   ✅ D2 Loss berubah-ubah (normal)")
            
        # G Loss stuck check
        g_stuck = all(loss == 20.0 for loss in g_losses)
        if g_stuck:
            print(f"   ❌ G Loss STUCK di 20.0 (clipping limit)")
        else:
            print(f"   ✅ G Loss berubah-ubah (normal)")
            
        # D1 Loss variation check
        d1_variation = max(d1_losses) - min(d1_losses)
        if d1_variation < 0.001:
            print(f"   ⚠️  D1 Loss variasi sangat kecil ({d1_variation:.6f})")
        else:
            print(f"   ✅ D1 Loss variasi normal ({d1_variation:.6f})")
            
    except FileNotFoundError:
        print("❌ File training log tidak ditemukan")
        return
        
    print(f"\n🔧 PERBAIKAN YANG DILAKUKAN:")
    print(f"   • D1 Loss clipping: 0-10 → 0-50")
    print(f"   • D2 Loss clipping: 0-20 → 0-100") 
    print(f"   • G Loss clipping: 0-20 → 0-100")
    print(f"   • Content loss weight: 5.0 → 1.0")
    print(f"   • Recognition loss weight: 10.0 → 2.0")
    
    print(f"\n💡 REKOMENDASI:")
    print(f"   1. Jalankan training ulang dengan konfigurasi yang sudah diperbaiki")
    print(f"   2. Monitor apakah loss values sekarang berubah-ubah")
    print(f"   3. Target loss ranges:")
    print(f"      • D1: 0.3-2.0 (discriminator visual)")
    print(f"      • D2: 0.5-15.0 (CRNN recognition)")
    print(f"      • G: 0.2-8.0 (generator)")

def test_configuration():
    """Berikan command test yang disarankan"""
    
    print(f"\n🧪 TEST COMMAND:")
    print(f"cd /home/lambda_one/tesis/GAN-HTR")
    print(f"poetry run python jnm_GAN_AHTR.py \\")
    print(f"    --epochs 5 \\")
    print(f"    --batch-size 8 \\")
    print(f"    --learning-rate 0.0001 \\")
    print(f"    --scenario S_iam_OP_debug \\")
    print(f"    --mode train")
    
    print(f"\n📊 HARAPAN SETELAH PERBAIKAN:")
    print(f"   • D1 Loss: ~0.5-1.5 (bervariasi)")
    print(f"   • D2 Loss: ~1.0-8.0 (bervariasi)")
    print(f"   • G Loss: ~0.8-5.0 (bervariasi)")
    print(f"   • Tidak ada nilai yang stuck di batas maksimum")

if __name__ == "__main__":
    analyze_training_logs()
    test_configuration()
