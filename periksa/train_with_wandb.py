#!/usr/bin/env python3
"""
Example script untuk menjalankan GAN-HTR training dengan WandB integration
Contoh penggunaan berbagai konfigurasi training
"""

import os
import sys
import subprocess
import argparse


def run_training_with_wandb(config_name, **kwargs):
    """
    Run training with specific configuration
    
    Args:
        config_name: Name of the configuration
        **kwargs: Additional arguments to pass to training script
    """
    
    print(f"🚀 Starting training with configuration: {config_name}")
    
    # Base command
    cmd = ["poetry", "run", "python", "jnm_GAN_AHTR.py"]
    
    # Add arguments
    for key, value in kwargs.items():
        if value is True:
            cmd.append(f"--{key.replace('_', '-')}")
        elif value is False or value is None:
            continue
        else:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"📋 Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Training completed successfully for {config_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for {config_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted for {config_name}")
        return False
    
    return True


def main():
    """Main function with predefined training configurations"""
    
    parser = argparse.ArgumentParser(description='Run GAN-HTR training with WandB')
    
    parser.add_argument('--config', type=str, 
                       choices=['quick-test', 'stable-training', 'long-training', 'custom'],
                       default='stable-training',
                       help='Training configuration to use')
    
    # Custom configuration options
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--wandb-project', type=str, help='WandB project name')
    parser.add_argument('--scenario', type=str, help='Training scenario name')
    
    args = parser.parse_args()
    
    print("🎯 GAN-HTR Training with WandB Integration")
    print(f"⚙️  Configuration: {args.config}")
    
    # Predefined configurations
    configs = {
        'quick-test': {
            'epochs': 5,
            'batch_size': 1,
            'learning_rate': 0.00001,
            'scenario': 'S_test_wandb',
            'wandb_project': 'gan-htr-quick-test',
            'save_interval': 2,
            'eval_interval': 1,
            'patience': 5
        },
        
        'stable-training': {
            'epochs': 20,
            'batch_size': 1,
            'learning_rate': 0.00001,
            'scenario': 'S_stable_wandb',
            'wandb_project': 'gan-htr-stable-training',
            'save_interval': 5,
            'eval_interval': 2,
            'patience': 10,
            'adv_weight': 0.5,
            'content_weight': 1.0,
            'recognition_weight': 0.5
        },
        
        'long-training': {
            'epochs': 50,
            'batch_size': 2,
            'learning_rate': 0.00005,
            'scenario': 'S_long_wandb',
            'wandb_project': 'gan-htr-long-training',
            'save_interval': 10,
            'eval_interval': 5,
            'patience': 15,
            'adv_weight': 0.7,
            'content_weight': 1.2,
            'recognition_weight': 0.8
        },
        
        'custom': {}
    }
    
    # Get base configuration
    if args.config in configs:
        config = configs[args.config].copy()
    else:
        config = {}
    
    # Override with custom arguments
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.wandb_project:
        config['wandb_project'] = args.wandb_project
    if args.scenario:
        config['scenario'] = args.scenario
    
    # Always enable WandB and images for these examples
    config['enable_wandb_images'] = True
    config['wandb_log_freq'] = 25
    
    print("\n📊 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Run training
    success = run_training_with_wandb(args.config, **config)
    
    if success:
        print(f"\n🎉 Training configuration '{args.config}' completed successfully!")
        if 'wandb_project' in config:
            print(f"📈 Check results at: https://wandb.ai/your-username/{config['wandb_project']}")
    else:
        print(f"\n❌ Training configuration '{args.config}' failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
