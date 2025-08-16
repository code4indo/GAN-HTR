#!/usr/bin/env python3
"""
Script untuk menjalankan diagnostic training GAN-HTR
Usage: poetry run python periksa/run_diagnostic.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_diagnostic import TrainingDiagnostic, test_ctc_loss_stability, create_emergency_training_config

def main():
    print("🧪 Running GAN-HTR Training Diagnostic...")
    
    # Test CTC loss stability
    ctc_stable = test_ctc_loss_stability()
    
    if not ctc_stable:
        print("❌ CTC loss is unstable. Applying emergency configuration...")
        config = create_emergency_training_config()
        print("🚨 Emergency config:")
        for key, value in config.items():
            print(f"   {key}: {value}")
    
    # Create diagnostic instance for testing
    diagnostic = TrainingDiagnostic()
    
    # Test with sample data
    import tensorflow as tf
    import numpy as np
    
    print("\n📊 Simulating training batches...")
    for i in range(20):
        # Simulate batch performance
        d1_loss = np.random.normal(1.5, 0.5)
        d2_loss = np.random.normal(100, 50) if i < 10 else np.random.normal(10, 2)  # Simulate improvement
        g_loss = np.random.normal(5, 2)
        batch_time = np.random.normal(0.8, 0.2)
        
        diagnostic.log_batch_performance(i+1, d1_loss, d2_loss, g_loss, batch_time, 4)
    
    # Get suggestions
    suggestions = diagnostic.suggest_fixes()
    if suggestions:
        print("\n💡 Training Suggestions:")
        for suggestion in suggestions:
            print(f"   {suggestion}")
    
    # Generate plot
    diagnostic.plot_training_progress()
    
    print("\n✅ Diagnostic completed successfully!")

if __name__ == "__main__":
    main()
