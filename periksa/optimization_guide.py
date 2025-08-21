#!/usr/bin/env python3
"""
Quick Start Hyperparameter Optimization untuk GAN-HTR
Script ini memberikan panduan step-by-step untuk optimization
"""

import os
import sys
import subprocess

def step_1_baseline():
    """Step 1: Establish baseline performance"""
    
    print("🎯 STEP 1: BASELINE TRAINING")
    print("="*50)
    
    print("Jalankan command ini untuk establish baseline:")
    print("\n📊 Baseline Training:")
    
    cmd = """poetry run python jnm_GAN_AHTR.py \\
    --epochs 15 \\
    --batch-size 2 \\
    --scenario "baseline_v1" \\
    --wandb-project "gan-htr-baseline" \\
    --enable-wandb-images \\
    --learning-rate 0.00001"""
    
    print(cmd)
    
    print("\n✅ Setelah selesai, catat metrics berikut:")
    print("   • Final val/g_loss")
    print("   • Training stability (no divergence)")
    print("   • Time per epoch")
    print("   • GPU memory usage")


def step_2_coarse_search():
    """Step 2: Coarse hyperparameter search"""
    
    print("\n🔍 STEP 2: COARSE HYPERPARAMETER SEARCH")
    print("="*50)
    
    print("Jalankan coarse search untuk explore parameter space:")
    
    cmd = """poetry run python periksa/start_sweep.py \\
    --project "gan-htr-coarse-search" \\
    --count 25"""
    
    print(cmd)
    
    print("\n📋 Parameter ranges yang akan di-explore:")
    print("   • Learning Rate: 1e-6 to 1e-3 (log-uniform)")
    print("   • Batch Size: [1, 2, 4]")
    print("   • ADV Weight: 0.1 to 2.0")
    print("   • Content Weight: 0.5 to 3.0")
    print("   • Recognition Weight: 0.1 to 1.0")
    
    print("\n⏱️  Expected time: 3-4 hours untuk 25 runs")
    print("💡 Monitor di: https://wandb.ai/your-username/gan-htr-coarse-search")


def step_3_analyze_results():
    """Step 3: Analyze coarse search results"""
    
    print("\n📊 STEP 3: ANALYZE COARSE SEARCH RESULTS")
    print("="*50)
    
    print("Di WandB dashboard, lakukan analysis berikut:")
    
    print("\n1. 🏆 Identify Top Performers:")
    print("   • Sort runs by 'val/g_loss' (ascending)")
    print("   • Take top 10 runs")
    print("   • Note their hyperparameter values")
    
    print("\n2. 📈 Parameter Patterns:")
    print("   • Use 'Parallel Coordinates' plot")
    print("   • Look for parameter correlations")
    print("   • Check 'Parameter Importance' plot")
    
    print("\n3. 🎯 Extract Optimal Ranges:")
    print("   • Learning Rate: [min dari top 10, max dari top 10]")
    print("   • Loss Weights: ratio patterns")
    print("   • Batch Size: most frequent dalam top performers")
    
    print("\n4. 🚨 Check for Issues:")
    print("   • Any runs yang diverged (very high loss)?")
    print("   • Training instability patterns?")
    print("   • GPU memory issues?")


def step_4_fine_tuning():
    """Step 4: Fine-tuning around best parameters"""
    
    print("\n🎯 STEP 4: FINE-TUNING OPTIMIZATION")
    print("="*50)
    
    print("Berdasarkan analysis Step 3, update parameter ranges:")
    
    print("\n📝 Update Fine-tuning Config:")
    print("1. Edit file: periksa/wandb_integration.py")
    print("2. Dalam function create_sweep_config(), update ranges:")
    print("   • 'learning-rate': {'min': <best_min>, 'max': <best_max>}")
    print("   • 'adv-weight': {'min': <best_min>, 'max': <best_max>}")
    print("   • dst...")
    
    print("\n🚀 Run Fine-tuning Sweep:")
    cmd = """poetry run python periksa/start_sweep.py \\
    --project "gan-htr-fine-tuning" \\
    --count 15"""
    
    print(cmd)
    
    print("\n⚙️  Settings untuk fine-tuning:")
    print("   • Epochs: 20 (longer untuk better evaluation)")
    print("   • Narrower parameter ranges")
    print("   • Focus on parameter interactions")


def step_5_production_validation():
    """Step 5: Production validation"""
    
    print("\n🏭 STEP 5: PRODUCTION VALIDATION")
    print("="*50)
    
    print("Test best hyperparameters dengan full training:")
    
    print("\n🏆 Get Best Parameters dari Fine-tuning:")
    print("   • Select #1 performer dari Step 4")
    print("   • Note exact hyperparameter values")
    
    print("\n🚀 Run Production Training:")
    cmd = """poetry run python jnm_GAN_AHTR.py \\
    --epochs 100 \\
    --batch-size <best_batch_size> \\
    --learning-rate <best_lr> \\
    --adv-weight <best_adv> \\
    --content-weight <best_content> \\
    --recognition-weight <best_recognition> \\
    --scenario "production_optimal" \\
    --wandb-project "gan-htr-production" \\
    --enable-wandb-images"""
    
    print(cmd)
    
    print("\n🔬 Validation Steps:")
    print("   • Run multiple seeds (3-5 runs)")
    print("   • Check consistency across runs")
    print("   • Validate pada different test sets")
    print("   • Compare dengan baseline")


def step_6_comparison_analysis():
    """Step 6: Final comparison and selection"""
    
    print("\n📊 STEP 6: FINAL COMPARISON & MODEL SELECTION")
    print("="*50)
    
    print("Compare all results untuk final model selection:")
    
    print("\n📈 Metrics Comparison:")
    print("   Baseline vs Optimized:")
    print("   • Validation Loss: {baseline} → {optimized}")
    print("   • Training Stability: {baseline} → {optimized}")
    print("   • Convergence Speed: {baseline} → {optimized}")
    print("   • Final Model Quality: {baseline} → {optimized}")
    
    print("\n🎯 Final Decision Criteria:")
    print("   • Lowest validation loss")
    print("   • Stable training (no divergence)")
    print("   • Consistent performance across seeds")
    print("   • Reasonable training time")
    print("   • Good generalization (train/val balance)")
    
    print("\n💾 Save Optimal Configuration:")
    print("   • Document best hyperparameters")
    print("   • Save model checkpoints")
    print("   • Create reproducible training script")


def quick_start_demo():
    """Jalankan quick demo untuk testing"""
    
    print("\n🚀 QUICK START DEMO")
    print("="*50)
    
    print("Untuk quick testing optimization workflow:")
    
    cmd = """poetry run python periksa/start_sweep.py \\
    --project "gan-htr-quick-demo" \\
    --count 5"""
    
    print(cmd)
    
    print("\n💡 Demo ini akan:")
    print("   • Run 5 quick experiments (simulation mode)")
    print("   • Show WandB integration working")
    print("   • Demonstrate sweep functionality")
    print("   • Take ~5-10 minutes")
    
    print("\n🔗 Monitor progress:")
    print("   https://wandb.ai/your-username/gan-htr-quick-demo")


def main():
    """Main workflow guide"""
    
    print("🎯 HYPERPARAMETER OPTIMIZATION WORKFLOW untuk GAN-HTR")
    print("="*70)
    
    print("\n📋 Complete Workflow Overview:")
    print("   Step 1: Baseline Training (30 min)")
    print("   Step 2: Coarse Search (3-4 hours)")
    print("   Step 3: Analyze Results (30 min)")
    print("   Step 4: Fine-tuning (2-3 hours)")
    print("   Step 5: Production Validation (8-12 hours)")
    print("   Step 6: Final Comparison (30 min)")
    print("   Total: ~15-20 hours untuk complete optimization")
    
    step_1_baseline()
    step_2_coarse_search()
    step_3_analyze_results()
    step_4_fine_tuning()
    step_5_production_validation()
    step_6_comparison_analysis()
    
    print("\n" + "="*70)
    print("🎯 QUICK START OPTIONS")
    print("="*70)
    
    quick_start_demo()
    
    print("\n💡 TIPS UNTUK SUCCESS:")
    print("   • Start dengan quick demo untuk validate setup")
    print("   • Monitor GPU temperature during long sweeps")
    print("   • Use tmux/screen untuk long-running processes")
    print("   • Check WandB dashboard regularly")
    print("   • Save intermediate results")
    
    print("\n🚨 COMMON ISSUES & SOLUTIONS:")
    print("   • GPU memory error → Reduce batch size")
    print("   • Training divergence → Lower learning rate")
    print("   • Slow convergence → Increase learning rate")
    print("   • Overfitting → Adjust loss weights")
    print("   • Network issues → Use WANDB_MODE='offline'")
    
    print("\n🎉 SUCCESS INDICATORS:")
    print("   • 30-50% reduction dalam validation loss")
    print("   • Stable training curves")
    print("   • Consistent results across multiple runs")
    print("   • Better visual quality dalam generated images")


if __name__ == "__main__":
    main()
