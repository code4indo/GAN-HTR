#!/usr/bin/env python3
"""
GAN-HTR Training Script untuk WandB Sweep
Script ini dirancang khusus untuk digunakan dengan WandB hyperparameter sweeps
"""

import os
import sys
import argparse
import wandb

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def sweep_train():
    """
    Training function yang dioptimalkan untuk WandB sweep
    """
    
    # Initialize WandB run FIRST
    wandb.init()
    
    # Get sweep configuration dari WandB
    config = wandb.config
    
    print(f"🔧 Sweep Run Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Build sys.argv dari WandB config untuk compatibility dengan argparse
    # Convert underscore to dash untuk argparse compatibility
    new_argv = ['sweep_train.py']
    
    for key, value in config.items():
        # Convert underscore to dash
        arg_name = key.replace('_', '-')
        new_argv.extend([f'--{arg_name}', str(value)])
    
    # Override sys.argv dengan config dari WandB
    original_argv = sys.argv[:]
    sys.argv = new_argv
    
    print(f"🔧 Generated argv: {sys.argv}")
    
    try:
        # Import after setting argv
        from jnm_GAN_AHTR import parse_arguments
        
        # Parse arguments dengan WandB config
        main_args = parse_arguments()
        
        # Set additional sweep-specific defaults
        main_args.disable_wandb = True  # Disable separate init since we already have one
        main_args.save_interval = 3     # Save more frequently for short runs
        main_args.eval_interval = 2     # Evaluate more frequently
        main_args.start_epoch = 0
        
        print(f"🚀 Starting sweep training with:")
        print(f"   Epochs: {main_args.epochs}")
        print(f"   Batch Size: {main_args.batch_size}")
        print(f"   Learning Rate: {main_args.learning_rate}")
        print(f"   Scenario: {main_args.scenario}")
        
        # SIMULATION MODE for testing (akan diganti dengan actual training)
        print("🧪 Running simulated training for sweep testing...")
        
        import time
        import random
        
        for epoch in range(main_args.epochs):
            time.sleep(1)  # Simulate training time (reduced untuk testing)
            
            # Simulate metrics yang respond terhadap hyperparameters
            lr_effect = main_args.learning_rate * 10000  # Lower LR = lower loss
            batch_effect = main_args.batch_size * 0.1     # Larger batch = slightly higher loss
            adv_effect = main_args.adv_weight * 0.1       # Higher adv weight = slight loss variation
            
            fake_d1_loss = max(0.1, 2.0 - lr_effect + batch_effect + random.uniform(-0.3, 0.3))
            fake_d2_loss = max(0.1, 5.0 - lr_effect + batch_effect + random.uniform(-0.5, 0.5))
            fake_g_loss = max(0.1, 3.0 - lr_effect + batch_effect + adv_effect + random.uniform(-0.4, 0.4))
            fake_val_loss = fake_g_loss + random.uniform(-0.2, 0.2)
            
            # Log metrics ke WandB
            wandb.log({
                'epoch': epoch,
                'train/d1_loss': fake_d1_loss,
                'train/d2_loss': fake_d2_loss, 
                'train/g_loss': fake_g_loss,
                'val/g_loss': fake_val_loss,
                'train/learning_rate': main_args.learning_rate,
                'performance/batch_size': main_args.batch_size,
                'config/adv_weight': main_args.adv_weight,
                'config/content_weight': main_args.content_weight,
                'config/recognition_weight': main_args.recognition_weight
            })
            
            print(f"Epoch {epoch}: D1={fake_d1_loss:.3f}, D2={fake_d2_loss:.3f}, G={fake_g_loss:.3f}, Val={fake_val_loss:.3f}")
        
        # Final metric for optimization
        final_val_loss = fake_val_loss
        wandb.log({'final_val_loss': final_val_loss})
        
        print(f"✅ Sweep run completed with final validation loss: {final_val_loss:.4f}")
        
        # TODO: Replace simulation dengan actual training:
        # from jnm_GAN_AHTR import train_GAN_crnn
        # train_GAN_crnn(main_args.epochs, main_args.batch_size)
        
    except Exception as e:
        print(f"❌ Sweep run failed: {e}")
        import traceback
        traceback.print_exc()
        wandb.log({"training_failed": True, "error": str(e)})
        raise e
    
    finally:
        # Restore original argv
        sys.argv = original_argv
        # WandB run akan otomatis finish oleh agent


if __name__ == "__main__":
    sweep_train()
