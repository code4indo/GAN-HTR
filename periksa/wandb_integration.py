"""
Enhanced WandB Integration for GAN-HTR Project
Terintegrasi dengan sistem monitoring existing TrainingDiagnostic dan DynamicTrainingMonitor
"""

import wandb
import os
import time
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
from typing import Dict, Any, List, Optional

# Import existing monitoring systems
from .training_diagnostic import TrainingDiagnostic
from .training_monitor import DynamicTrainingMonitor


class WANDBGANIntegration:
    """
    Advanced WandB integration specifically designed for GAN-HTR training
    Integrates seamlessly with existing monitoring systems
    """
    
    def __init__(self, 
                 project_name: str = "gan-htr-thesis",
                 run_name: Optional[str] = None,
                 config: Optional[Dict] = None,
                 enable_image_logging: bool = True,
                 log_frequency: int = 10):
        """
        Initialize WandB integration with enhanced configuration
        
        Args:
            project_name: WandB project name
            run_name: Specific run name (auto-generated if None)
            config: Training configuration dictionary
            enable_image_logging: Whether to log sample images
            log_frequency: Log metrics every N batches
        """
        
        self.project_name = project_name
        self.run_name = run_name or f"gan-htr-{int(time.time())}"
        self.enable_image_logging = enable_image_logging
        self.log_frequency = log_frequency
        
        # Initialize existing monitoring systems
        self.diagnostic = TrainingDiagnostic()
        self.dynamic_monitor = None  # Will be set during training
        
        # Training state tracking
        self.batch_count = 0
        self.epoch_count = 0
        self.best_val_loss = float('inf')
        
        # Metrics accumulation
        self.batch_metrics = []
        self.epoch_metrics = []
        
        # Initialize WandB
        self._initialize_wandb(config)
        
        print(f"🎯 WandB Integration initialized for project: {project_name}")
        print(f"📊 Run name: {self.run_name}")
        print(f"🖼️  Image logging: {'Enabled' if enable_image_logging else 'Disabled'}")
        
    def _initialize_wandb(self, config: Optional[Dict] = None):
        """Initialize WandB with comprehensive configuration"""
        
        # Default configuration
        default_config = {
            "framework": "tensorflow",
            "architecture": "gan-htr",
            "model_type": "conditional_gan_with_crnn",
            "dataset": "nan_handwriting",
            "optimizer": "adam",
            "loss_functions": ["mse", "binary_crossentropy", "ctc"],
            "training_strategy": "distributed",
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        # Initialize WandB run
        try:
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=default_config,
                reinit=True,
                settings=wandb.Settings(start_method="fork")
            )
            
            # Log system information
            self._log_system_info()
            
            print("✅ WandB initialization successful")
            
        except Exception as e:
            print(f"⚠️ WandB initialization failed: {e}")
            print("🔄 Continuing without WandB logging...")
    
    def _log_system_info(self):
        """Log system and environment information"""
        try:
            system_info = {
                "gpu_count": len(tf.config.list_physical_devices('GPU')),
                "gpu_names": [gpu.name for gpu in tf.config.list_physical_devices('GPU')],
                "tensorflow_version": tf.__version__,
                "python_version": f"{tf.python.version.VERSION}",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            wandb.config.update({"system_info": system_info})
            
        except Exception as e:
            print(f"⚠️ Could not log system info: {e}")
    
    def start_epoch(self, epoch: int, dynamic_monitor: DynamicTrainingMonitor = None):
        """
        Start tracking for new epoch
        
        Args:
            epoch: Current epoch number
            dynamic_monitor: Dynamic training monitor instance
        """
        self.epoch_count = epoch
        self.batch_count = 0
        self.batch_metrics = []
        
        if dynamic_monitor:
            self.dynamic_monitor = dynamic_monitor
        
        print(f"📊 WandB: Starting epoch {epoch}")
    
    def log_batch_metrics(self, 
                         epoch: int,
                         batch: int, 
                         d1_loss: float, 
                         d2_loss: float, 
                         g_loss: float, 
                         batch_time: float,
                         batch_size: int,
                         learning_rate: float = None,
                         additional_metrics: Dict = None):
        """
        Enhanced batch metrics logging with comprehensive tracking
        
        Args:
            epoch: Current epoch
            batch: Batch number
            d1_loss: Discriminator 1 (adversarial) loss
            d2_loss: Discriminator 2 (CRNN) loss  
            g_loss: Generator loss
            batch_time: Time taken for batch processing
            batch_size: Batch size
            learning_rate: Current learning rate
            additional_metrics: Any additional metrics to log
        """
        
        self.batch_count += 1
        
        # Calculate derived metrics
        samples_per_second = batch_size / batch_time if batch_time > 0 else 0
        
        # Prepare metrics dictionary
        metrics = {
            "epoch": epoch,
            "batch": batch,
            "train/d1_loss": float(d1_loss),
            "train/d2_loss": float(d2_loss), 
            "train/g_loss": float(g_loss),
            "train/total_loss": float(d1_loss + d2_loss + g_loss),
            "performance/batch_time": batch_time,
            "performance/samples_per_second": samples_per_second,
            "performance/throughput": samples_per_second,
            "step": epoch * 1000 + batch,  # Global step counter
        }
        
        # Add learning rate if provided
        if learning_rate is not None:
            metrics["train/learning_rate"] = learning_rate
        
        # Add additional metrics
        if additional_metrics:
            for key, value in additional_metrics.items():
                metrics[f"additional/{key}"] = value
        
        # Log to existing diagnostic system
        self.diagnostic.log_batch_performance(batch, d1_loss, d2_loss, g_loss, batch_time, batch_size)
        
        # Store for epoch aggregation
        self.batch_metrics.append(metrics)
        
        # Log to WandB at specified frequency
        if self.batch_count % self.log_frequency == 0:
            try:
                wandb.log(metrics)
                
                # Check for alerts and anomalies
                self._check_training_alerts(d1_loss, d2_loss, g_loss, epoch, batch)
                
            except Exception as e:
                print(f"⚠️ WandB batch logging failed: {e}")
    
    def _check_training_alerts(self, d1_loss: float, d2_loss: float, g_loss: float, epoch: int, batch: int):
        """Check for training anomalies and send WandB alerts"""
        
        alerts = []
        
        # Check for loss explosion
        if g_loss > 50.0:
            alerts.append({
                "title": "Generator Loss Explosion",
                "text": f"Generator loss exploded to {g_loss:.2f} at epoch {epoch}, batch {batch}",
                "level": "ERROR"
            })
        
        if d2_loss > 100.0:
            alerts.append({
                "title": "CRNN Loss Explosion", 
                "text": f"CRNN (D2) loss exploded to {d2_loss:.2f} at epoch {epoch}, batch {batch}",
                "level": "ERROR"
            })
        
        # Check for discriminator collapse
        if d1_loss < 0.01:
            alerts.append({
                "title": "Discriminator Collapse",
                "text": f"D1 loss very low ({d1_loss:.4f}) - potential discriminator collapse",
                "level": "WARN"
            })
        
        # Check for generator collapse
        if g_loss < 0.01:
            alerts.append({
                "title": "Generator Collapse",
                "text": f"Generator loss very low ({g_loss:.4f}) - potential mode collapse",
                "level": "WARN"
            })
        
        # Send alerts to WandB
        for alert in alerts:
            try:
                wandb.alert(
                    title=alert["title"],
                    text=alert["text"],
                    level=getattr(wandb.AlertLevel, alert["level"], wandb.AlertLevel.WARN)
                )
            except Exception as e:
                print(f"⚠️ Could not send WandB alert: {e}")
    
    def log_epoch_summary(self, 
                         epoch: int,
                         train_metrics: Dict,
                         val_metrics: Dict = None,
                         model_metrics: Dict = None):
        """
        Log comprehensive epoch summary
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics (avg losses, etc.)
            val_metrics: Validation metrics
            model_metrics: Model-specific metrics (weights, gradients, etc.)
        """
        
        # Calculate epoch aggregations from batch metrics
        if self.batch_metrics:
            epoch_summary = {
                "epoch": epoch,
                "train/epoch_avg_d1_loss": np.mean([m["train/d1_loss"] for m in self.batch_metrics]),
                "train/epoch_avg_d2_loss": np.mean([m["train/d2_loss"] for m in self.batch_metrics]),
                "train/epoch_avg_g_loss": np.mean([m["train/g_loss"] for m in self.batch_metrics]),
                "train/epoch_avg_total_loss": np.mean([m["train/total_loss"] for m in self.batch_metrics]),
                "performance/epoch_avg_throughput": np.mean([m["performance/samples_per_second"] for m in self.batch_metrics]),
                "performance/epoch_total_batches": len(self.batch_metrics),
            }
        else:
            epoch_summary = {"epoch": epoch}
        
        # Add provided training metrics
        if train_metrics:
            for key, value in train_metrics.items():
                epoch_summary[f"train/epoch_{key}"] = value
        
        # Add validation metrics
        if val_metrics:
            for key, value in val_metrics.items():
                epoch_summary[f"val/{key}"] = value
                
        # Add model metrics
        if model_metrics:
            for key, value in model_metrics.items():
                epoch_summary[f"model/{key}"] = value
        
        # Store epoch metrics
        self.epoch_metrics.append(epoch_summary)
        
        # Log to WandB
        try:
            wandb.log(epoch_summary)
            
            # Update best validation loss tracking
            if val_metrics and 'g_loss' in val_metrics:
                if val_metrics['g_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['g_loss']
                    epoch_summary['val/is_best'] = True
                    wandb.log({"val/best_loss": self.best_val_loss})
                else:
                    epoch_summary['val/is_best'] = False
                    
        except Exception as e:
            print(f"⚠️ WandB epoch logging failed: {e}")
        
        print(f"📊 Epoch {epoch} summary logged to WandB")
    
    def log_sample_images(self,
                         epoch: int,
                         original_images: List[np.ndarray],
                         degraded_images: List[np.ndarray], 
                         enhanced_images: List[np.ndarray],
                         max_images: int = 4):
        """
        Log sample images for visual progress tracking
        
        Args:
            epoch: Current epoch
            original_images: List of original (ground truth) images
            degraded_images: List of degraded input images
            enhanced_images: List of enhanced output images  
            max_images: Maximum number of images to log
        """
        
        if not self.enable_image_logging:
            return
        
        try:
            wandb_images = []
            
            for i in range(min(len(original_images), max_images)):
                # Ensure images are in correct format (0-1 range)
                orig = self._normalize_image(original_images[i])
                deg = self._normalize_image(degraded_images[i])
                enh = self._normalize_image(enhanced_images[i])
                
                # Create comparison image
                comparison = np.hstack([deg, enh, orig])
                
                wandb_images.append(
                    wandb.Image(
                        comparison,
                        caption=f"Epoch {epoch} Sample {i+1}: Degraded | Enhanced | Original"
                    )
                )
            
            # Log images to WandB
            wandb.log({
                f"samples/epoch_{epoch}": wandb_images,
                f"samples/epoch": epoch
            })
            
            print(f"🖼️ {len(wandb_images)} sample images logged for epoch {epoch}")
            
        except Exception as e:
            print(f"⚠️ Image logging failed: {e}")
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range for WandB"""
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        elif img.max() > 1.0:
            return img / img.max()
        else:
            return img.astype(np.float32)
    
    def log_model_artifacts(self,
                           epoch: int,
                           generator,
                           discriminator_1, 
                           discriminator_2,
                           save_path: str):
        """
        Log model artifacts and checkpoints to WandB
        
        Args:
            epoch: Current epoch
            generator: Generator model
            discriminator_1: Discriminator 1 model
            discriminator_2: Discriminator 2 (CRNN) model
            save_path: Local path where models are saved
        """
        
        try:
            # Create artifact for this epoch
            artifact = wandb.Artifact(
                name=f"gan-htr-models-epoch-{epoch}",
                type="model",
                description=f"GAN-HTR models at epoch {epoch}"
            )
            
            # Add model files to artifact
            if os.path.exists(os.path.join(save_path, "generator.weights.h5")):
                artifact.add_file(os.path.join(save_path, "generator.weights.h5"))
            
            if os.path.exists(os.path.join(save_path, "discriminator.weights.h5")):
                artifact.add_file(os.path.join(save_path, "discriminator.weights.h5"))
            
            if os.path.exists(os.path.join(save_path, "rcnn.weights.h5")):
                artifact.add_file(os.path.join(save_path, "rcnn.weights.h5"))
            
            if os.path.exists(os.path.join(save_path, "gan.weights.h5")):
                artifact.add_file(os.path.join(save_path, "gan.weights.h5"))
            
            # Add metadata
            if os.path.exists(os.path.join(save_path, "metadata.json")):
                artifact.add_file(os.path.join(save_path, "metadata.json"))
            
            # Log artifact
            wandb.log_artifact(artifact)
            
            print(f"📦 Model artifacts logged for epoch {epoch}")
            
        except Exception as e:
            print(f"⚠️ Model artifact logging failed: {e}")
    
    def log_hyperparameter_sweep_result(self, metric_value: float, metric_name: str = "val_g_loss"):
        """Log result for hyperparameter sweep"""
        try:
            wandb.log({f"sweep/{metric_name}": metric_value})
        except Exception as e:
            print(f"⚠️ Sweep result logging failed: {e}")
    
    def log_training_diagnostics(self):
        """Log diagnostic information from existing monitoring systems"""
        
        if self.diagnostic:
            try:
                suggestions = self.diagnostic.suggest_fixes()
                if suggestions:
                    # Log diagnostic suggestions as text
                    wandb.log({
                        "diagnostics/suggestions_count": len(suggestions),
                        "diagnostics/latest_suggestions": suggestions[:5]  # Top 5
                    })
            except Exception as e:
                print(f"⚠️ Diagnostic logging failed: {e}")
        
        if self.dynamic_monitor:
            try:
                recommendations = self.dynamic_monitor.get_recommendations()
                if recommendations:
                    wandb.log({
                        "monitor/recommendations_count": len(recommendations),
                        "monitor/latest_recommendations": recommendations[:5]
                    })
            except Exception as e:
                print(f"⚠️ Monitor logging failed: {e}")
    
    def log_custom_metrics(self, metrics: Dict[str, Any], prefix: str = "custom"):
        """Log custom metrics with optional prefix"""
        try:
            prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            wandb.log(prefixed_metrics)
        except Exception as e:
            print(f"⚠️ Custom metrics logging failed: {e}")
    
    def finish_run(self, summary_metrics: Dict = None):
        """
        Finish WandB run with optional summary
        
        Args:
            summary_metrics: Final summary metrics to log
        """
        
        try:
            # Log final summary
            if summary_metrics:
                for key, value in summary_metrics.items():
                    wandb.run.summary[key] = value
            
            # Log training completion
            wandb.run.summary["training_completed"] = True
            wandb.run.summary["total_epochs"] = self.epoch_count
            wandb.run.summary["best_val_loss"] = self.best_val_loss
            
            # Finish run
            wandb.finish()
            
            print("🎯 WandB run completed successfully")
            
        except Exception as e:
            print(f"⚠️ WandB run completion failed: {e}")


class WANDBHyperparameterSweep:
    """
    WandB Hyperparameter Sweep Configuration for GAN-HTR
    """
    
    @staticmethod
    def create_sweep_config(project_name: str = "gan-htr-sweeps") -> Dict:
        """
        Create comprehensive sweep configuration for GAN-HTR hyperparameter optimization
        
        Returns:
            Dictionary with sweep configuration
        """
        
        sweep_config = {
            'method': 'bayes',  # Bayesian optimization
            'metric': {
                'name': 'val/g_loss',
                'goal': 'minimize'
            },
            # PERBAIKAN: Specify program to run
            'program': 'sweep_train.py',
            'command': [
                '${env}',
                'python',
                '${program}',
                '${args}'
            ],
            'parameters': {
                # Learning rates
                'learning-rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-6,
                    'max': 1e-3
                },
                
                # Batch size
                'batch-size': {
                    'values': [1, 2, 4]  # Reduced for stability
                },
                
                # Loss weights
                'adv-weight': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 2.0
                },
                
                'content-weight': {
                    'distribution': 'uniform', 
                    'min': 0.5,
                    'max': 3.0
                },
                
                'recognition-weight': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 1.0
                },
                
                # Optimization parameters
                'patience': {
                    'values': [5, 8, 10, 15]
                },
                
                # Fixed parameters for sweep stability
                'epochs': {
                    'value': 5  # Short epochs for quick sweep testing
                },
                
                'scenario': {
                    'value': 'S_sweep_test'
                }
            },
            
            # Early termination for poor runs
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 3,
                'eta': 2
            }
        }
        
        return sweep_config
    
    @staticmethod
    def start_sweep(project_name: str = "gan-htr-sweeps") -> str:
        """
        Start hyperparameter sweep
        
        Returns:
            Sweep ID for running agents
        """
        
        sweep_config = WANDBHyperparameterSweep.create_sweep_config(project_name)
        
        try:
            sweep_id = wandb.sweep(sweep_config, project=project_name)
            print(f"🚀 Hyperparameter sweep started: {sweep_id}")
            print(f"📊 Project: {project_name}")
            print("💡 Run sweep agent with:")
            print(f"   poetry run wandb agent {sweep_id}")
            
            return sweep_id
            
        except Exception as e:
            print(f"❌ Sweep creation failed: {e}")
            return None


def create_wandb_config_from_args(args) -> Dict:
    """
    Create WandB configuration from command line arguments
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary for WandB
    """
    
    config = {
        # Training parameters
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "start_epoch": args.start_epoch,
        "learning_rate": args.learning_rate,
        
        # Model parameters
        "scenario": args.scenario,
        "database_path": args.database_path,
        
        # Training configuration
        "patience": args.patience,
        "min_delta": args.min_delta,
        "save_interval": args.save_interval,
        "eval_interval": args.eval_interval,
        
        # Loss weights
        "adv_weight": args.adv_weight,
        "content_weight": args.content_weight,
        "recognition_weight": args.recognition_weight,
        
        # System configuration
        "gpu_devices": args.gpu_devices,
        "mode": args.mode,
        
        # Model architecture
        "max_text_length": 128,
        "img_width": 1024,
        "img_height": 128,
        "input_size_crnn": [1024, 128, 1],
        "input_size_gan": [128, 1024, 1],
    }
    
    return config


# Example usage and integration helper functions
def setup_wandb_for_training(args, run_name: str = None) -> WANDBGANIntegration:
    """
    Setup WandB integration for training session
    
    Args:
        args: Command line arguments
        run_name: Optional custom run name
        
    Returns:
        Configured WANDBGANIntegration instance
    """
    
    # Create configuration from args
    config = create_wandb_config_from_args(args)
    
    # Create integration instance
    wandb_integration = WANDBGANIntegration(
        project_name=f"gan-htr-{args.scenario}",
        run_name=run_name,
        config=config,
        enable_image_logging=True,
        log_frequency=25  # Log every 25 batches
    )
    
    return wandb_integration


if __name__ == "__main__":
    # Example of creating a hyperparameter sweep
    print("🔧 Creating example hyperparameter sweep...")
    
    sweep_id = WANDBHyperparameterSweep.start_sweep("gan-htr-test-sweeps")
    if sweep_id:
        print(f"✅ Sweep created successfully: {sweep_id}")
    else:
        print("❌ Sweep creation failed")
