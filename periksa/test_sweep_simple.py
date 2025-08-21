#!/usr/bin/env python3
"""
Test Script untuk WandB Sweep
Script sederhana untuk test sweep configuration tanpa full training
"""

import wandb
import time
import random
import numpy as np

def main():
    """Simple test function untuk sweep"""
    
    # Initialize WandB
    wandb.init()
    
    # Get configuration from sweep
    config = wandb.config
    
    print(f"🧪 Test Run with config:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Simulate training dengan random metrics
    epochs = config.get('epochs', 10)
    
    for epoch in range(epochs):
        # Simulate some computation
        time.sleep(1)
        
        # Generate fake metrics berdasarkan config
        lr = config.get('learning_rate', 0.001)
        batch_size = config.get('batch_size', 1)
        
        # Fake loss yang responds to hyperparameters
        base_loss = 5.0
        lr_effect = np.log(lr * 1000000)  # Lower LR = lower loss (roughly)
        batch_effect = batch_size * 0.1   # Larger batch = slightly higher loss
        noise = random.uniform(-0.5, 0.5)
        
        fake_loss = max(0.1, base_loss + lr_effect + batch_effect + noise)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train/loss': fake_loss,
            'val/g_loss': fake_loss + random.uniform(-0.2, 0.2),
            'learning_rate': lr,
            'batch_size': batch_size
        })
        
        print(f"Epoch {epoch}: loss = {fake_loss:.4f}")
    
    # Final metric for sweep optimization
    final_loss = fake_loss
    wandb.log({'final_loss': final_loss})
    
    print(f"✅ Test completed with final loss: {final_loss:.4f}")
    
    # Finish run
    wandb.finish()

if __name__ == "__main__":
    main()
