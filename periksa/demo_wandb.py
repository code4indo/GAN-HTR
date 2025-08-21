#!/usr/bin/env python3
"""
WandB Integration Demo Script
Demonstrasi lengkap fitur WandB untuk GAN-HTR
"""

import os
import sys
import time

# Add parent directory untuk imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def demo_basic_integration():
    """Demo basic WandB integration"""
    
    print("🎬 DEMO: Basic WandB Integration")
    print("="*50)
    
    print("\n1. 📊 Training dengan WandB monitoring:")
    print("   poetry run python jnm_GAN_AHTR.py --epochs 5 --batch-size 1 --scenario 'demo_basic'")
    
    print("\n2. 🖼️  Training dengan image logging:")
    print("   poetry run python jnm_GAN_AHTR.py --epochs 5 --enable-wandb-images --scenario 'demo_images'")
    
    print("\n3. 🚫 Training tanpa WandB (local testing):")
    print("   poetry run python jnm_GAN_AHTR.py --epochs 5 --disable-wandb --scenario 'demo_local'")


def demo_hyperparameter_sweep():
    """Demo hyperparameter optimization"""
    
    print("\n🎬 DEMO: Hyperparameter Optimization")
    print("="*50)
    
    print("\n1. 🔬 Quick sweep untuk testing (simulasi):")
    print("   poetry run python periksa/start_sweep.py --project 'demo-quick-sweep' --count 3")
    
    print("\n2. 🏭 Production sweep dengan actual training:")
    print("   # Edit sweep config untuk production parameters")
    print("   # Uncomment actual training call di sweep_production.py")
    print("   poetry run python periksa/start_sweep.py --project 'demo-production' --count 10")
    
    print("\n3. 👥 Manual sweep agent:")
    print("   poetry run wandb agent <sweep-id>")


def demo_monitoring_features():
    """Demo monitoring capabilities"""
    
    print("\n🎬 DEMO: Monitoring Features")
    print("="*50)
    
    print("\n📈 Tracked Metrics:")
    print("   • train/d1_loss, train/d2_loss, train/g_loss")
    print("   • val/g_loss, val/accuracy")
    print("   • performance/epoch_time, performance/gpu_memory")
    print("   • config/adv_weight, config/content_weight")
    
    print("\n🖼️  Image Logging:")
    print("   • Input (degraded) images")
    print("   • Ground truth images") 
    print("   • Generated/enhanced images")
    print("   • Side-by-side comparisons")
    
    print("\n⚙️  Configuration Tracking:")
    print("   • All hyperparameters")
    print("   • Model architecture details")
    print("   • Training strategy parameters")


def demo_best_practices():
    """Demo best practices"""
    
    print("\n🎬 DEMO: Best Practices")
    print("="*50)
    
    print("\n🎯 Project Organization:")
    print("   gan-htr-baseline     # Baseline experiments")
    print("   gan-htr-ablation     # Ablation studies")
    print("   gan-htr-optimization # Hyperparameter sweeps")
    print("   gan-htr-production   # Production runs")
    
    print("\n🏷️  Run Naming:")
    print("   gan-htr-20250117-baseline-v1")
    print("   gan-htr-20250117-ablation-loss-weights")
    print("   gan-htr-20250117-sweep-best-params")
    
    print("\n⚡ Performance Tips:")
    print("   • Use --wandb-log-freq 50 untuk large datasets")
    print("   • Enable --enable-wandb-images hanya untuk important runs")
    print("   • Start dengan small sweeps (count=5) untuk validation")


def demo_troubleshooting():
    """Demo troubleshooting scenarios"""
    
    print("\n🎬 DEMO: Troubleshooting")
    print("="*50)
    
    print("\n🔑 Authentication:")
    print("   poetry run wandb login")
    print("   # Get API key dari: https://wandb.ai/authorize")
    
    print("\n🌐 Network Issues:")
    print("   export WANDB_MODE='offline'")
    print("   # Train offline, sync later dengan:")
    print("   poetry run wandb sync wandb/run-{timestamp}")
    
    print("\n🔧 GPU Memory Issues:")
    print("   # Disable image logging untuk large batches:")
    print("   python jnm_GAN_AHTR.py --batch-size 8  # Without --enable-wandb-images")
    
    print("\n🐛 Debug Mode:")
    print("   # Check logs:")
    print("   tail -f wandb/debug-internal.log")


def run_quick_demo():
    """Run a quick actual demo"""
    
    print("\n🚀 RUNNING QUICK DEMO")
    print("="*50)
    
    print("\n🔬 Starting quick hyperparameter sweep demo...")
    print("   This will run 2 simulated training runs dengan different hyperparameters")
    
    import subprocess
    
    try:
        # Start sweep
        cmd = [
            "poetry", "run", "python", "periksa/start_sweep.py",
            "--project", "gan-htr-demo-final",
            "--count", "2"
        ]
        
        print(f"\n💻 Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/lambda_one/tesis/GAN-HTR")
        
        if result.returncode == 0:
            print("\n✅ Demo completed successfully!")
            print("\n📊 Results available at:")
            print("   https://wandb.ai/jatnikonm-binus-university/gan-htr-demo-final")
        else:
            print(f"\n❌ Demo failed: {result.stderr}")
            
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


def main():
    """Main demo function"""
    
    print("🎭 WandB Integration Demo untuk GAN-HTR")
    print("="*60)
    
    demo_basic_integration()
    demo_hyperparameter_sweep()
    demo_monitoring_features()
    demo_best_practices() 
    demo_troubleshooting()
    
    print("\n" + "="*60)
    print("🎯 Demo Script Summary:")
    print("   • Basic integration examples")
    print("   • Hyperparameter optimization setup")
    print("   • Monitoring capabilities overview")
    print("   • Best practices guidelines")
    print("   • Troubleshooting common issues")
    
    # Ask if user wants to run quick demo
    print("\n🤔 Would you like to run a quick live demo? (y/n)")
    try:
        response = input().strip().lower()
        if response in ['y', 'yes']:
            run_quick_demo()
        else:
            print("\n👋 Demo completed! Happy training with WandB!")
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Happy training!")


if __name__ == "__main__":
    main()
