# 🎯 WandB Integration untuk GAN-HTR

Dokumentasi lengkap untuk integrasi Weights & Biases (WandB) dengan proyek GAN-HTR.

## 📋 Overview

WandB integration telah ditambahkan ke proyek GAN-HTR untuk memberikan:
- Real-time monitoring training progress
- Advanced visualization dan dashboards
- Hyperparameter optimization dengan sweeps
- Model artifacts management
- Alert system untuk training anomalies
- Team collaboration dan experiment tracking

## 🚀 Quick Start

### 1. Setup WandB Account
```bash
# Install sudah dilakukan via poetry
# Login ke WandB account
poetry run wandb login
```

### 2. Basic Training dengan WandB
```bash
# Training dengan konfigurasi default + WandB
poetry run python jnm_GAN_AHTR.py --epochs 10 --wandb-project "my-gan-htr-project"

# Training dengan custom configuration
poetry run python jnm_GAN_AHTR.py \
  --epochs 20 \
  --batch-size 1 \
  --learning-rate 0.00001 \
  --scenario "S_wandb_test" \
  --wandb-project "gan-htr-thesis" \
  --wandb-run-name "test-run-001" \
  --enable-wandb-images \
  --wandb-log-freq 10
```

### 3. Using Predefined Configurations
```bash
# Quick test (5 epochs)
poetry run python periksa/train_with_wandb.py --config quick-test

# Stable training (20 epochs)
poetry run python periksa/train_with_wandb.py --config stable-training

# Long training (50 epochs)
poetry run python periksa/train_with_wandb.py --config long-training

# Custom configuration
poetry run python periksa/train_with_wandb.py --config custom \
  --epochs 30 --batch-size 2 --wandb-project "my-project"
```

## 📊 Features yang Dilog

### Batch-level Metrics
- **Loss metrics**: D1 loss, D2 loss (CRNN), Generator loss, Total loss
- **Performance metrics**: Batch time, Samples per second, Throughput
- **Training metrics**: Learning rate, Batch number, Epoch

### Epoch-level Metrics  
- **Aggregated losses**: Average losses across all batches
- **Validation metrics**: Validation loss, Best validation loss tracking
- **Training diagnostics**: Recommendations dari monitoring systems
- **Model metrics**: Learning rate changes, Patience counter

### Visual Tracking
- **Sample images**: Comparison degraded → enhanced → original
- **Training progress**: Loss curves, Performance graphs
- **Model artifacts**: Saved checkpoints dan weights

### Alerts & Monitoring
- **Loss explosion detection**: Alert saat loss > threshold
- **Training instability**: Detection untuk discriminator/generator collapse
- **Performance degradation**: Speed monitoring dan alerts

## 🔧 Configuration Options

### Command Line Arguments
```bash
# WandB specific options
--wandb-project          # WandB project name
--wandb-run-name         # Specific run name  
--disable-wandb          # Disable WandB logging
--wandb-log-freq         # Log frequency (default: 25 batches)
--enable-wandb-images    # Enable image logging (default: True)

# Training parameters
--epochs                 # Number of epochs
--batch-size            # Batch size
--learning-rate         # Learning rate
--scenario              # Training scenario name

# Advanced options
--patience              # Early stopping patience
--save-interval         # Save model every N epochs
--eval-interval         # Run evaluation every N epochs
--adv-weight           # Adversarial loss weight
--content-weight       # Content loss weight  
--recognition-weight   # Recognition loss weight
```

### Programmatic Configuration
```python
from periksa.wandb_integration import WANDBGANIntegration, create_wandb_config_from_args

# Create WandB integration
wandb_integration = WANDBGANIntegration(
    project_name="my-gan-htr-project",
    run_name="experiment-001",
    config={
        "epochs": 20,
        "batch_size": 1,
        "learning_rate": 0.00001,
        # ... other config
    },
    enable_image_logging=True,
    log_frequency=25
)
```

## 🔍 Hyperparameter Sweeps

### 1. Create and Run Sweep
```bash
# Create sweep dan run otomatis
poetry run python periksa/start_sweep.py --project "gan-htr-sweeps" --count 10

# Hanya create sweep (manual agent execution)
poetry run python periksa/start_sweep.py --project "gan-htr-sweeps" --create-only
```

### 2. Manual Sweep Agent
```bash
# Jika sweep sudah dibuat
poetry run wandb agent <sweep-id>
```

### 3. Sweep Configuration
Default sweep mengoptimalkan:
- **Learning rate**: 1e-6 to 1e-3 (log uniform)
- **Batch size**: [1, 2, 4, 8]
- **Loss weights**: Adversarial, Content, Recognition
- **Training parameters**: Patience, Dropout rate, LR reduction factor

## 📈 Dashboard & Visualization

### Key Dashboards di WandB:
1. **Training Overview**: Real-time loss curves, performance metrics
2. **Sample Images**: Visual progress comparison
3. **System Metrics**: GPU utilization, memory usage
4. **Hyperparameter Comparison**: Multi-run analysis
5. **Model Artifacts**: Checkpoint management

### Custom Visualizations:
- Loss trend analysis dengan gradient information
- Training speed optimization tracking
- Early stopping analysis
- Discriminator vs Generator balance monitoring

## 🚨 Alert System

### Automatic Alerts untuk:
- **Generator Loss Explosion** (> 50.0)
- **CRNN Loss Explosion** (> 100.0)  
- **Discriminator Collapse** (loss < 0.01)
- **Generator Collapse** (loss < 0.01)
- **Training Instability** (NaN/Inf values)

### Alert Configuration:
```python
# Custom alert thresholds
wandb_integration._check_training_alerts(d1_loss, d2_loss, g_loss, epoch, batch)
```

## 📦 Model Artifacts

### Automatic Logging:
- **Model weights**: Generator, Discriminator 1, Discriminator 2, GAN
- **Training metadata**: Configuration, metrics, timestamps
- **Checkpoints**: Regular interval saves dengan versioning

### Artifact Management:
```python
# Manual artifact logging
wandb_integration.log_model_artifacts(
    epoch=epoch,
    generator=generator,
    discriminator_1=discriminator_1, 
    discriminator_2=discriminator_2,
    save_path=checkpoint_path
)
```

## 🔧 Integration dengan Existing Systems

### Compatibility dengan:
- **TrainingDiagnostic**: Suggestions dan recommendations
- **DynamicTrainingMonitor**: Advanced monitoring decisions  
- **Emergency Training**: Fallback mechanisms
- **Existing Logging**: JSON logs, CSV exports

### Enhanced Features:
- **Dual logging**: WandB + existing file-based logs
- **Error handling**: Graceful fallback jika WandB fails
- **Performance optimization**: Minimal overhead pada training

## 💡 Best Practices

### 1. Naming Conventions
```bash
# Project naming
gan-htr-{scenario}         # Main experiments
gan-htr-{scenario}-sweeps  # Hyperparameter sweeps
gan-htr-debug             # Development/debugging

# Run naming  
{scenario}-{date}-{version}   # e.g., stable-20240817-v1
```

### 2. Logging Frequency
- **Development**: Log every 10 batches
- **Production**: Log every 25-50 batches  
- **Long training**: Log every 100 batches

### 3. Resource Management
```bash
# Untuk training panjang
--wandb-log-freq 50       # Reduce logging frequency
--disable-wandb-images    # Disable images untuk save bandwidth
```

## 🐛 Troubleshooting

### Common Issues:

#### 1. WandB Login Issues
```bash
# Re-login
poetry run wandb login --relogin

# Check status
poetry run wandb status
```

#### 2. Project Access Issues
```bash
# Check project exists
poetry run wandb project list

# Create new project
poetry run wandb project create my-project
```

#### 3. Upload Failures
```bash
# Check internet connection
# Reduce logging frequency
--wandb-log-freq 100

# Disable images temporarily  
--disable-wandb-images
```

#### 4. Memory Issues dengan Images
```python
# Limit image samples
max_images=2  # Instead of default 4

# Reduce image frequency
if epoch % 5 == 0:  # Only every 5 epochs
    log_sample_images(...)
```

## 📚 Examples

### Example 1: Quick Development Test
```bash
poetry run python jnm_GAN_AHTR.py \
  --epochs 3 \
  --batch-size 1 \
  --scenario "S_dev_test" \
  --wandb-project "gan-htr-development" \
  --wandb-log-freq 5 \
  --eval-interval 1
```

### Example 2: Production Training
```bash
poetry run python jnm_GAN_AHTR.py \
  --epochs 50 \
  --batch-size 2 \
  --learning-rate 0.00005 \
  --scenario "S_production_v1" \
  --wandb-project "gan-htr-thesis-final" \
  --wandb-run-name "final-model-v1" \
  --save-interval 10 \
  --eval-interval 5 \
  --patience 15
```

### Example 3: Hyperparameter Sweep
```bash
# Start comprehensive sweep
poetry run python periksa/start_sweep.py \
  --project "gan-htr-optimization" \
  --count 20
```

## 📊 Monitoring Dashboard

### Key Metrics to Watch:
1. **Training Stability**: Loss curves should be smooth
2. **Convergence**: Validation loss should decrease over time
3. **Performance**: Throughput should remain consistent
4. **Resource Usage**: GPU memory and utilization
5. **Image Quality**: Visual improvement over epochs

### Red Flags:
- Oscillating losses (unstable training)
- Flat loss curves (learning rate too low)
- Exponential loss increase (learning rate too high)
- Decreasing throughput (memory leaks)

## 🎯 Results Analysis

Setelah training, gunakan WandB dashboard untuk:
1. **Compare experiments**: Multi-run comparison
2. **Analyze hyperparameters**: Correlation analysis
3. **Export results**: Download untuk paper/thesis
4. **Share findings**: Team collaboration features

---

**✅ Integration telah siap digunakan!**

Untuk memulai, jalankan:
```bash
poetry run python periksa/train_with_wandb.py --config quick-test
```
