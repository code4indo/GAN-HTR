#!/usr/bin/env python3
"""
Production WandB Sweep Script untuk GAN-HTR
Script ini menjalankan actual training (bukan simulasi)
"""

import os
import sys
import wandb

# Add parent directory untuk imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sweep_train():
    """
    Production training function untuk WandB sweep
    """
    
    # Initialize WandB run FIRST
    wandb.init()
    
    # Get sweep configuration dari WandB
    config = wandb.config
    
    print(f"🔧 Production Sweep Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Build sys.argv dari WandB config untuk compatibility dengan argparse
    new_argv = ['sweep_production.py']
    
    for key, value in config.items():
        # Convert underscore to dash untuk argparse compatibility
        arg_name = key.replace('_', '-')
        new_argv.extend([f'--{arg_name}', str(value)])
    
    # Override sys.argv dengan config dari WandB
    original_argv = sys.argv[:]
    sys.argv = new_argv
    
    print(f"🔧 Generated argv: {sys.argv}")
    
    try:
        # Import actual training function
        from jnm_GAN_AHTR import parse_arguments, train_gan
        
        # Parse arguments dengan WandB config
        main_args = parse_arguments()
        
        # Set production-specific configurations
        main_args.disable_wandb = True  # Disable separate init since we already have one
        main_args.save_interval = 5     # Save every 5 epochs untuk production
        main_args.eval_interval = 3     # Evaluate every 3 epochs
        main_args.start_epoch = 0
        
        print(f"🚀 Starting production sweep training with:")
        print(f"   Epochs: {main_args.epochs}")
        print(f"   Batch Size: {main_args.batch_size}")
        print(f"   Learning Rate: {main_args.learning_rate}")
        print(f"   Scenario: {main_args.scenario}")
        print(f"   ADV Weight: {main_args.adv_weight}")
        print(f"   Content Weight: {main_args.content_weight}")
        print(f"   Recognition Weight: {main_args.recognition_weight}")
        
        # RUN ACTUAL TRAINING
        print("🏭 Running ACTUAL GAN-HTR training...")
        
        # Call the actual training function
        train_gan(
            epochs=main_args.epochs,
            batch_size=main_args.batch_size,
            start_epoch=main_args.start_epoch,
            scenario=main_args.scenario,
            learning_rate=main_args.learning_rate,
            gpu_devices=main_args.gpu_devices,
            database_path=main_args.database_path,
            resume=main_args.resume,
            resume_epoch=main_args.resume_epoch,
            mode=main_args.mode,
            patience=main_args.patience,
            min_delta=main_args.min_delta,
            save_interval=main_args.save_interval,
            eval_interval=main_args.eval_interval,
            adv_weight=main_args.adv_weight,
            content_weight=main_args.content_weight,
            recognition_weight=main_args.recognition_weight,
            wandb_project=main_args.wandb_project,
            wandb_run_name=main_args.wandb_run_name,
            disable_wandb=main_args.disable_wandb,
            wandb_log_freq=main_args.wandb_log_freq,
            enable_wandb_images=main_args.enable_wandb_images
        )
        
        print(f"✅ Production sweep run completed successfully!")
        
    except Exception as e:
        print(f"❌ Production sweep run failed: {e}")
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
