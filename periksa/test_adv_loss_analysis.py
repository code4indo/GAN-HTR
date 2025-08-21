import numpy as np
import matplotlib.pyplot as plt

def analyze_adv_loss_stagnation(training_history):
    """
    Analisis apakah adversarial loss mengalami stagnasi
    """
    adv_losses = training_history['train_g_loss_adv']
    
    # Hitung variasi loss dalam window terakhir
    window_size = 10
    if len(adv_losses) >= window_size:
        recent_losses = adv_losses[-window_size:]
        variance = np.var(recent_losses)
        mean_loss = np.mean(recent_losses)
        
        print(f"Recent {window_size} epochs analysis:")
        print(f"Mean adv_loss: {mean_loss:.6f}")
        print(f"Variance: {variance:.8f}")
        print(f"Standard deviation: {np.sqrt(variance):.6f}")
        
        # Deteksi stagnasi
        if variance < 1e-6:
            print("⚠️  STAGNATION DETECTED: Adversarial loss variance very low")
            
            if mean_loss > 0.5:
                print("💡 Suggestion: Generator too weak - consider:")
                print("   - Reduce discriminator learning rate")
                print("   - Increase generator learning rate")
                print("   - Adjust adv_weight")
            else:
                print("💡 Suggestion: Generator too strong - consider:")
                print("   - Increase discriminator learning rate") 
                print("   - Reduce generator learning rate")
                print("   - Add noise to discriminator training")
        
        # Plot trend
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(adv_losses)
        plt.title('Adversarial Loss Trend')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(recent_losses)
        plt.title(f'Recent {window_size} Epochs')
        plt.xlabel('Recent Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('/home/lambda_one/tesis/GAN-HTR/periksa/adv_loss_analysis.png')
        plt.show()

def suggest_training_adjustments(mean_adv_loss):
    """
    Berikan saran penyesuaian training berdasarkan nilai adv_loss
    """
    print("\n🔧 TRAINING ADJUSTMENT SUGGESTIONS:")
    
    if mean_adv_loss > 0.8:
        print("❌ Generator very weak:")
        print("   - Discriminator learning rate: 0.0001 → 0.00005")
        print("   - Generator learning rate: 0.0002 → 0.0003")
        print("   - Add discriminator noise/dropout")
        
    elif mean_adv_loss > 0.5:
        print("⚠️  Generator somewhat weak:")
        print("   - Adjust adv_weight: current → current * 0.8")
        print("   - Consider discriminator regularization")
        
    elif mean_adv_loss < 0.1:
        print("❌ Generator too strong:")
        print("   - Generator learning rate: current → current * 0.8") 
        print("   - Discriminator learning rate: current → current * 1.2")
        print("   - Reduce adv_weight")
        
    elif mean_adv_loss < 0.3:
        print("⚠️  Generator somewhat strong:")
        print("   - Monitor discriminator loss trend")
        print("   - Consider balancing loss weights")
    
    else:
        print("✅ Adversarial loss in reasonable range")
        print("   - Focus on gradient flow analysis")
        print("   - Check for cyclic patterns")

# Contoh penggunaan
if __name__ == "__main__":
    # Simulasi data untuk testing
    # Ganti dengan data aktual dari training
    sample_history = {
        'train_g_loss_adv': [0.7, 0.65, 0.6, 0.58, 0.55, 0.53, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52]
    }
    
    analyze_adv_loss_stagnation(sample_history)
    suggest_training_adjustments(0.52)