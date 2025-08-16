import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple, Optional

class DynamicTrainingMonitor:
    """Advanced training monitor with early intervention strategies"""
    
    def __init__(self, 
                 patience_epochs: int = 10,
                 min_improvement: float = 0.001,
                 max_loss_threshold: float = 100.0,
                 speed_threshold: float = 5.0,
                 save_dir: str = "training_logs"):
        
        self.patience_epochs = patience_epochs
        self.min_improvement = min_improvement
        self.max_loss_threshold = max_loss_threshold
        self.speed_threshold = speed_threshold
        self.save_dir = save_dir
        
        # Training history
        self.loss_history = {
            'epoch': [],
            'd1_loss': [],
            'd2_loss': [],
            'g_loss': [],
            'val_loss': [],
            'speed': [],
            'timestamp': []
        }
        
        # Control flags
        self.should_stop = False
        self.should_reduce_lr = False
        self.should_restart = False
        
        # Counters
        self.no_improvement_count = 0
        self.high_loss_count = 0
        self.slow_speed_count = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def update(self, epoch: int, d1_loss: float, d2_loss: float, 
              g_loss: float, val_loss: float, speed: float) -> Dict[str, bool]:
        """Update training metrics and return control decisions"""
        
        current_time = time.time()
        
        # Record metrics
        self.loss_history['epoch'].append(epoch)
        self.loss_history['d1_loss'].append(float(d1_loss))
        self.loss_history['d2_loss'].append(float(d2_loss))
        self.loss_history['g_loss'].append(float(g_loss))
        self.loss_history['val_loss'].append(float(val_loss))
        self.loss_history['speed'].append(float(speed))
        self.loss_history['timestamp'].append(current_time)
        
        # Analyze trends
        decisions = self._analyze_training_state(epoch)
        
        # Save log
        self._save_log()
        
        return decisions
    
    def _analyze_training_state(self, epoch: int) -> Dict[str, bool]:
        """Analyze current training state and make decisions"""
        
        decisions = {
            'stop_training': False,
            'reduce_lr': False,
            'restart_training': False,
            'adjust_batch_size': False,
            'change_strategy': False
        }
        
        if len(self.loss_history['val_loss']) < 3:
            return decisions
        
        recent_val_losses = self.loss_history['val_loss'][-5:]
        recent_g_losses = self.loss_history['g_loss'][-5:]
        recent_speeds = self.loss_history['speed'][-3:]
        
        # Check for improvement
        if len(recent_val_losses) >= 2:
            improvement = recent_val_losses[-2] - recent_val_losses[-1]
            if improvement < self.min_improvement:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0
        
        # Check for extremely high losses (divergence)
        if any(loss > self.max_loss_threshold for loss in recent_g_losses):
            self.high_loss_count += 1
        else:
            self.high_loss_count = 0
        
        # Check for slow training speed
        avg_speed = np.mean(recent_speeds)
        if avg_speed < self.speed_threshold:
            self.slow_speed_count += 1
        else:
            self.slow_speed_count = 0
        
        # Decision logic
        if self.high_loss_count >= 3:
            print(f"🚨 CRITICAL: Loss divergence detected! Suggesting restart with different hyperparameters")
            decisions['restart_training'] = True
            decisions['change_strategy'] = True
        
        elif self.no_improvement_count >= self.patience_epochs:
            if epoch < 50:  # Early in training
                print(f"⚠️ No improvement for {self.patience_epochs} epochs (early training). Reducing LR")
                decisions['reduce_lr'] = True
                self.no_improvement_count = 0  # Reset counter
            else:
                print(f"🛑 No improvement for {self.patience_epochs} epochs (late training). Stopping")
                decisions['stop_training'] = True
        
        elif self.slow_speed_count >= 3:
            print(f"🐌 Slow training speed detected. Suggesting batch size adjustment")
            decisions['adjust_batch_size'] = True
        
        return decisions
    
    def _save_log(self):
        """Save training log to file"""
        log_file = os.path.join(self.save_dir, f"training_log_{int(time.time())}.json")
        with open(log_file, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
    
    def get_recommendations(self) -> List[str]:
        """Get training recommendations based on current state"""
        recommendations = []
        
        if len(self.loss_history['g_loss']) < 5:
            return ["Continue training - insufficient data for analysis"]
        
        recent_losses = self.loss_history['g_loss'][-5:]
        
        if np.mean(recent_losses) > 50:
            recommendations.extend([
                "🔥 URGENT: Losses are extremely high",
                "💡 Reduce learning rate by factor of 10",
                "💡 Check data preprocessing and normalization",
                "💡 Consider warm-up training strategy",
                "💡 Verify loss function implementation"
            ])
        
        if self.slow_speed_count > 0:
            recommendations.extend([
                "⚡ Training speed is suboptimal",
                "💡 Reduce batch size or increase GPU memory efficiency",
                "💡 Check data loading pipeline bottlenecks",
                "💡 Enable mixed precision if not already active"
            ])
        
        return recommendations if recommendations else ["✅ Training appears stable"]

def create_emergency_training_config() -> Dict:
    """Create emergency training configuration for problematic cases"""
    return {
        'generator_lr': 1e-5,  # Much lower
        'discriminator_lr': 1e-5,  # Much lower
        'batch_size': 6,  # Smaller batch
        'warmup_epochs': 5,
        'loss_weights': [0.1, 1.0, 1.0],  # Reduce adversarial weight
        'patience': 15,
        'monitor_every_n_batches': 10
    }
