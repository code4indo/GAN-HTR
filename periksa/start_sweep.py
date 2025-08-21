#!/usr/bin/env python3
"""
WandB Hyperparameter Sweep Script for GAN-HTR
Runs automated hyperparameter optimization
"""

import os
import sys
import argparse
import wandb

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from periksa.wandb_integration import WANDBHyperparameterSweep


def main():
    """Main function to start hyperparameter sweep"""
    
    parser = argparse.ArgumentParser(description='Start WandB Hyperparameter Sweep for GAN-HTR')
    
    parser.add_argument('--project', type=str, default='gan-htr-sweeps',
                       help='WandB project name for sweeps (default: gan-htr-sweeps)')
    parser.add_argument('--count', type=int, default=10,
                       help='Number of sweep runs to execute (default: 10)')
    parser.add_argument('--create-only', action='store_true',
                       help='Only create sweep, don\'t run agent')
    
    args = parser.parse_args()
    
    print("🚀 GAN-HTR Hyperparameter Sweep Setup")
    print(f"📊 Project: {args.project}")
    print(f"🔢 Planned runs: {args.count}")
    
    # Create sweep
    print("\n🔧 Creating hyperparameter sweep configuration...")
    sweep_id = WANDBHyperparameterSweep.start_sweep(args.project)
    
    if not sweep_id:
        print("❌ Failed to create sweep")
        sys.exit(1)
    
    print(f"✅ Sweep created successfully: {sweep_id}")
    
    if args.create_only:
        print("\n📋 Sweep created. Run agents manually with:")
        print(f"   poetry run wandb agent {sweep_id}")
        print("\n💡 Or use this script to run agents:")
        print(f"   poetry run python periksa/start_sweep.py --project {args.project}")
        return
    
    # Run sweep agent
    print(f"\n🤖 Starting sweep agent for {args.count} runs...")
    print("⚠️  Note: Each run will train the full GAN-HTR model")
    print("⏱️  This may take several hours depending on your configuration")
    
    try:
        wandb.agent(sweep_id, count=args.count)
        print("🎉 Sweep completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Sweep interrupted by user")
        print(f"💾 Partial results available in WandB project: {args.project}")
        
    except Exception as e:
        print(f"❌ Sweep failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
