# WandB Integration untuk GAN-HTR

## 📖 Overview

Integrasi ini menambahkan monitoring dan hyperparameter optimization yang comprehensive untuk proyek GAN-HTR menggunakan Weights & Biases (WandB).

## 🚀 Features

### 1. **Training Monitoring**
- Real-time loss tracking (Generator, Discriminator 1, Discriminator 2)
- Validation metrics monitoring
- Learning rate scheduling visualization
- Training diagnostics dan performance metrics
- System resource monitoring

### 2. **Image Logging**
- Input images (degraded documents)
- Ground truth images
- Generated/enhanced images
- Side-by-side comparisons
- Training progress visualization

### 3. **Hyperparameter Optimization**
- Automated Bayesian optimization
- Grid search dan random search
- Early termination untuk efisiensi
- Multi-run comparison
- Best hyperparameter discovery

### 4. **Model Versioning**
- Automatic model artifact tracking
- Checkpoint management
- Run comparison dan analysis
- Configuration versioning

## 📦 Installation

```bash
# Install WandB via Poetry
poetry add wandb

# Login ke WandB account
poetry run wandb login
```

## 🔧 Basic Usage

### Training dengan WandB Monitoring

```bash
# Training dengan WandB integration
poetry run python jnm_GAN_AHTR.py --epochs 50 --batch-size 4 --scenario "MyExperiment" --enable-wandb-images

# Training dengan custom WandB project
poetry run python jnm_GAN_AHTR.py --wandb-project "my-gan-project" --wandb-run-name "experiment-v1"

# Disable WandB untuk local testing
poetry run python jnm_GAN_AHTR.py --disable-wandb
```

### Hyperparameter Optimization

```bash
# Start hyperparameter sweep (simulasi untuk testing)
poetry run python periksa/start_sweep.py --project "gan-htr-optimization" --count 10

# Start production sweep dengan actual training
poetry run python periksa/start_sweep.py --project "gan-htr-production" --count 20

# Monitor sweep progress
poetry run wandb agent <sweep-id>
```

## 📊 Monitored Metrics

### Training Metrics
- `train/d1_loss`: Discriminator 1 loss
- `train/d2_loss`: Discriminator 2 (CRNN) loss  
- `train/g_loss`: Generator loss
- `train/combined_loss`: Combined generator loss
- `train/learning_rate`: Current learning rate

### Validation Metrics
- `val/g_loss`: Validation generator loss
- `val/accuracy`: Validation accuracy
- `val/perplexity`: Validation perplexity

### Performance Metrics
- `performance/epoch_time`: Time per epoch
- `performance/batch_time`: Time per batch
- `performance/gpu_memory`: GPU memory usage
- `performance/batch_size`: Current batch size

### Configuration Tracking
- `config/adv_weight`: Adversarial loss weight
- `config/content_weight`: Content loss weight
- `config/recognition_weight`: Recognition loss weight

## 🎛️ Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--wandb-project` | `gan-htr-{scenario}` | WandB project name |
| `--wandb-run-name` | Auto-generated | Custom run name |
| `--disable-wandb` | `False` | Disable WandB logging |
| `--wandb-log-freq` | `25` | Logging frequency (batches) |
| `--enable-wandb-images` | `False` | Enable image logging |

### Environment Variables

```bash
# Set WandB API key
export WANDB_API_KEY="your_api_key_here"

# Set WandB mode
export WANDB_MODE="online"  # or "offline" for local logging

# Set default project
export WANDB_PROJECT="gan-htr-default"
```

## 🔍 Hyperparameter Sweep Configuration

### Optimized Parameters

1. **Learning Rate**: `1e-6` to `1e-3` (log-uniform)
2. **Batch Size**: `[1, 2, 4]`
3. **Loss Weights**:
   - Adversarial: `0.1` to `2.0`
   - Content: `0.5` to `3.0`
   - Recognition: `0.1` to `1.0`
4. **Training Strategy**:
   - Patience: `[5, 8, 10, 15]`
   - Epochs: `5` (for quick testing)

### Sweep Methods
- **Bayes**: Bayesian optimization (recommended)
- **Grid**: Exhaustive grid search
- **Random**: Random parameter sampling

### Early Termination
- **Hyperband**: Automatically stops poor-performing runs
- **Min iterations**: 3 epochs minimum
- **Eta**: 2 (aggressive pruning)

## 📁 File Structure

```
periksa/
├── wandb_integration.py       # Core WandB integration classes
├── start_sweep.py             # Hyperparameter sweep starter
├── sweep_production.py        # Production sweep script
└── README_wandb.md           # This documentation

wandb/                         # WandB local storage
├── run-{timestamp}/          # Individual run data
└── debug-internal.log        # Debug logs
```

## 🔄 Integration Points

### Main Training Script (`jnm_GAN_AHTR.py`)
- Automatic initialization di awal training
- Batch-level dan epoch-level logging
- Model checkpoint tracking
- Error handling dan graceful shutdown

### Core Integration (`wandb_integration.py`)
- `WANDBGANIntegration`: Main training monitoring
- `WANDBHyperparameterSweep`: Sweep configuration
- Helper functions untuk setup dan configuration

## 🎯 Best Practices

### 1. **Naming Convention**
```python
# Project naming
project_name = f"gan-htr-{scenario_name}"

# Run naming  
run_name = f"gan-htr-{timestamp}"
```

### 2. **Resource Management**
```python
# Enable image logging hanya untuk important runs
--enable-wandb-images

# Adjust logging frequency untuk performance
--wandb-log-freq 50  # For large datasets
```

### 3. **Sweep Strategy**
```python
# Start dengan quick sweep untuk testing
epochs: 5

# Scale up untuk production
epochs: 50
```

### 4. **Monitoring Setup**
```python
# Use different projects untuk different experiments
--wandb-project "gan-htr-baseline"
--wandb-project "gan-htr-ablation"
--wandb-project "gan-htr-production"
```

## 🚨 Troubleshooting

### Common Issues

**1. Authentication Error**
```bash
poetry run wandb login
# Enter API key dari https://wandb.ai/authorize
```

**2. Sweep Agent Not Running**
```bash
# Check sweep status
poetry run wandb sweep --help

# Manual agent start
poetry run wandb agent <sweep-id>
```

**3. GPU Memory Issues dengan Image Logging**
```bash
# Disable image logging untuk large batches
python jnm_GAN_AHTR.py --batch-size 8  # Without --enable-wandb-images
```

**4. Network Issues**
```bash
# Use offline mode
export WANDB_MODE="offline"

# Sync later
poetry run wandb sync wandb/run-{timestamp}
```

### Performance Optimization

**1. Reduce Logging Frequency**
```bash
--wandb-log-freq 100  # Log every 100 batches instead of 25
```

**2. Selective Image Logging**
```python
# Only log images every N epochs
if epoch % 5 == 0:
    wandb_integration.log_images(...)
```

**3. Batch Size untuk Sweeps**
```python
# Use smaller batch sizes untuk quick exploration
'batch-size': {'values': [1, 2]}  # Instead of [1, 2, 4, 8]
```

## 📈 Advanced Usage

### Custom Metrics
```python
# Log custom metrics
wandb_integration.log_custom_metrics({
    'custom/metric1': value1,
    'custom/metric2': value2
})
```

### Model Artifacts
```python
# Save model as artifact
wandb_integration.save_model_artifact(
    model_path='checkpoints/best_model.h5',
    artifact_name='gan-htr-best-model'
)
```

### Comparison Analysis
```python
# Compare multiple runs
wandb_integration.compare_runs([
    'run_id_1', 'run_id_2', 'run_id_3'
])
```

## 🎯 Example Commands

```bash
# Basic training dengan monitoring
poetry run python jnm_GAN_AHTR.py --epochs 20 --batch-size 2 --scenario "baseline_test"

# Advanced training dengan full logging
poetry run python jnm_GAN_AHTR.py \
    --epochs 50 \
    --batch-size 4 \
    --scenario "production_v1" \
    --wandb-project "gan-htr-production" \
    --wandb-run-name "exp-001" \
    --enable-wandb-images \
    --wandb-log-freq 25

# Quick hyperparameter sweep
poetry run python periksa/start_sweep.py \
    --project "gan-htr-quick-test" \
    --count 5

# Production hyperparameter optimization
poetry run python periksa/start_sweep.py \
    --project "gan-htr-optimization" \
    --count 50
```

## 🤝 Integration Status

✅ **Completed Features:**
- Basic training monitoring
- Loss tracking dan visualization
- Hyperparameter sweep setup
- Image logging capability
- Configuration management
- Error handling

🔄 **In Progress:**
- Advanced metrics correlation
- Automated model selection
- Multi-GPU optimization tracking

🎯 **Future Enhancements:**
- Real-time training dashboards
- Automated report generation
- Integration dengan model serving
- Advanced artifact management

---

## 📞 Support

Untuk questions atau issues:
1. Check WandB documentation: https://docs.wandb.ai/
2. Review error logs di `wandb/debug-internal.log`
3. Test dengan `--disable-wandb` untuk isolate issues
4. Use simulation mode untuk quick testing

**Happy Training! 🚀**
