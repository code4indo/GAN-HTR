#!/usr/bin/env python3
"""
Simple Sweep Test Script
Test WandB sweep dengan script sederhana
"""

import os
import sys
import argparse
import wandb

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from periksa.wandb_integration import WANDBHyperparameterSweep


def create_simple_sweep_config(project_name: str = "gan-htr-test-sweeps") -> dict:
    """Create simple sweep config for testing"""
    
    sweep_config = {
        'method': 'grid',  # Simple grid search for testing
        'metric': {
            'name': 'final_loss',
            'goal': 'minimize'
        },
        'program': 'test_sweep_simple.py',
        'parameters': {
            'learning_rate': {
                'values': [0.001, 0.0001, 0.00001]
            },
            'batch_size': {
                'values': [1, 2, 4]
            },
            'epochs': {
                'value': 3  # Short for testing
            }
        }
    }
    
    return sweep_config


def main():
    """Main function untuk test sweep"""
    
    parser = argparse.ArgumentParser(description='Test WandB Sweep for GAN-HTR')
    
    parser.add_argument('--project', type=str, default='gan-htr-test-sweeps',
                       help='WandB project name for test sweeps')
    parser.add_argument('--count', type=int, default=3,
                       help='Number of test runs (default: 3)')
    
    args = parser.parse_args()
    
    print("🧪 GAN-HTR Simple Sweep Test")
    print(f"📊 Project: {args.project}")
    print(f"🔢 Test runs: {args.count}")
    
    # Create simple sweep config
    sweep_config = create_simple_sweep_config(args.project)
    
    print("\n🔧 Creating test sweep...")
    try:
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(f"✅ Test sweep created: {sweep_id}")
        print(f"🌐 Sweep URL: https://wandb.ai/your-username/{args.project}/sweeps/{sweep_id}")
        
        print(f"\n🤖 Running {args.count} test agents...")
        wandb.agent(sweep_id, count=args.count)
        
        print("🎉 Test sweep completed successfully!")
        print(f"📈 Check results at: https://wandb.ai/your-username/{args.project}")
        
    except Exception as e:
        print(f"❌ Test sweep failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
