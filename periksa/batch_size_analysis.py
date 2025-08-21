#!/usr/bin/env python3
"""
Batch Size Analysis untuk GAN-HTR
Experiment script untuk test different batch sizes
"""

import os
import sys

def analyze_batch_size_impact():
    """
    Analisis impact batch size pada GAN-HTR training
    """
    
    print("🔬 BATCH SIZE IMPACT ANALYSIS untuk GAN-HTR")
    print("="*60)
    
    print("\n📊 THEORETICAL ANALYSIS:")
    
    print("\n🟢 BATCH SIZE = 1 (Current Default):")
    print("   ✅ Pros:")
    print("      • Maximum exploration (high variance)")
    print("      • Less memory usage")
    print("      • Better escape dari local minima")
    print("      • Good untuk initial training")
    print("   ❌ Cons:")
    print("      • Very noisy gradients") 
    print("      • Unstable training")
    print("      • Slower convergence")
    print("      • Poor GPU utilization")
    
    print("\n🟡 BATCH SIZE = 4:")
    print("   ✅ Pros:")
    print("      • Balanced noise vs stability")
    print("      • Better GPU utilization")
    print("      • More stable than batch=1")
    print("      • Good exploration still")
    print("   ❌ Cons:")
    print("      • 4x memory usage")
    print("      • Slightly less exploration")
    
    print("\n🟠 BATCH SIZE = 8:")
    print("   ✅ Pros:")
    print("      • Good stability")
    print("      • Efficient GPU usage")
    print("      • Smoother loss curves")
    print("   ❌ Cons:")
    print("      • 8x memory usage")
    print("      • Risk of overfitting")
    print("      • May miss fine details")
    
    print("\n🔴 BATCH SIZE = 16+:")
    print("   ✅ Pros:")
    print("      • Very stable training")
    print("      • Maximum GPU efficiency")
    print("   ❌ Cons:")
    print("      • High memory requirements")
    print("      • Poor generalization")
    print("      • Risk of mode collapse")
    print("      • Sharp minima convergence")
    
    print("\n" + "="*60)
    print("🎯 RECOMMENDATIONS untuk GAN-HTR")
    print("="*60)
    
    print("\n🏆 OPTIMAL STRATEGY:")
    print("   1. Start dengan batch=1 untuk exploration")
    print("   2. Increase ke batch=2 jika stable")  
    print("   3. Try batch=4 untuk better efficiency")
    print("   4. Monitor loss curves carefully")
    print("   5. Never go above batch=8 untuk GAN")
    
    print("\n🧪 EXPERIMENTAL COMMANDS:")
    print("\n   # Test batch=1 (current default)")
    print("   poetry run python jnm_GAN_AHTR.py --epochs 5 --batch-size 1 --scenario 'batch_test_1'")
    
    print("\n   # Test batch=2")
    print("   poetry run python jnm_GAN_AHTR.py --epochs 5 --batch-size 2 --scenario 'batch_test_2'")
    
    print("\n   # Test batch=4")
    print("   poetry run python jnm_GAN_AHTR.py --epochs 5 --batch-size 4 --scenario 'batch_test_4'")
    
    print("\n🔍 METRICS TO COMPARE:")
    print("   • Training stability (loss variance)")
    print("   • Convergence speed")
    print("   • Final validation loss")
    print("   • Generated image quality")
    print("   • GPU memory usage")
    print("   • Training time per epoch")
    
    print("\n⚠️  MEMORY CONSIDERATIONS:")
    print("   • Batch=1: ~4GB GPU memory")
    print("   • Batch=2: ~6GB GPU memory") 
    print("   • Batch=4: ~10GB GPU memory")
    print("   • Batch=8: ~18GB GPU memory (may OOM)")
    
    print("\n💡 SPECIAL TECHNIQUES:")
    print("   • Gradient Accumulation: Simulate larger batches")
    print("   • Progressive Batch Scaling: Start small, increase")
    print("   • Adaptive Batch Size: Change based on stability")


def demo_batch_size_sweep():
    """
    Demo untuk batch size optimization menggunakan WandB sweep
    """
    
    print("\n🎬 DEMO: BATCH SIZE OPTIMIZATION dengan WandB")
    print("="*60)
    
    print("\nSweep configuration untuk batch size optimization:")
    
    sweep_config = {
        'name': 'gan-htr-batch-size-optimization',
        'method': 'grid',  # Test all combinations
        'metric': {
            'name': 'val/g_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'batch-size': {
                'values': [1, 2, 4]  # Safe range
            },
            'learning-rate': {
                'value': 0.00001  # Fixed untuk fair comparison
            },
            'epochs': {
                'value': 10
            },
            'scenario': {
                'values': ['batch_test_1', 'batch_test_2', 'batch_test_4']
            }
        }
    }
    
    print("\n📊 This sweep akan test:")
    print("   • Batch Size: [1, 2, 4]")
    print("   • Fixed LR: 0.00001")
    print("   • Fixed Epochs: 10")
    print("   • Total runs: 3")
    
    print("\n🚀 To run batch size optimization:")
    print("   # Create specialized sweep untuk batch size")
    print("   # Edit periksa/wandb_integration.py dengan config di atas")
    print("   poetry run python periksa/start_sweep.py --project 'gan-htr-batch-optimization' --count 3")


if __name__ == "__main__":
    analyze_batch_size_impact()
    demo_batch_size_sweep()
