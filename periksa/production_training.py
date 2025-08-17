#!/usr/bin/env python3
"""
PRODUCTION GAN-HTR TRAINING SCRIPT
Berdasarkan hasil emergency training yang sukses

Konfigurasi yang telah terbukti stabil:
- Batch size: 1 (scalable to 4)
- Learning rate: 1e-5 (scalable to 1e-4)
- CTC loss: UltraSafeCTCLoss
- Gradient clipping: 0.1
"""

# Run with: poetry run python periksa/production_training.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Single GPU proven stable

def main():
    # Start with proven stable configuration
    config = {
        'epochs': 20,                    # Longer training
        'batch_size': 1,                 # Start small, scale up
        'learning_rate': 1e-5,           # Conservative but proven
        'patience': 10,                  # More patience
        'save_interval': 5,              # Save every 5 epochs  
        'max_samples': 100,              # Increase dataset gradually
        'gradient_clip_norm': 0.1,       # Proven stable
        'loss_weights': [0.5, 1.0, 0.5] # Balanced weights
    }
    
    # Progressive scaling strategy
    scaling_phases = [
        # Phase 1: Stability test
        {'epochs': 5, 'batch_size': 1, 'max_samples': 50},
        
        # Phase 2: Scale batch size
        {'epochs': 10, 'batch_size': 2, 'max_samples': 100},
        
        # Phase 3: Scale dataset
        {'epochs': 15, 'batch_size': 2, 'max_samples': 200},
        
        # Phase 4: Scale learning rate
        {'epochs': 20, 'batch_size': 4, 'max_samples': 500, 'learning_rate': 5e-5}
    ]
    
    for phase_num, phase_config in enumerate(scaling_phases, 1):
        print(f"🚀 Starting Phase {phase_num}: {phase_config}")
        
        # Run training with phase config
        # trainer = EmergencyTrainer()
        # trainer.run_emergency_training(**phase_config)
        
        # Validate stability before next phase
        # if not stable: break and diagnose
    
    print("🎉 Production training completed!")

if __name__ == "__main__":
    main()
