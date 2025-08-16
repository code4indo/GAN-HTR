"""
Fix Training Issues for GAN-HTR
Utility script to diagnose and fix common training problems
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from glob import glob
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainingIssuesFixer:
    def __init__(self, root_path='./', scenario='S_iam_OP'):
        self.root_path = root_path
        self.scenario = scenario
        self.results_path = os.path.join(root_path, f"ResultGan{scenario}")
        
    def check_gpu_setup(self):
        """Check GPU configuration and memory"""
        print("🔍 Checking GPU setup...")
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            print("❌ No GPUs detected!")
            return False
            
        print(f"✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
            
        # Check memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth enabled for all GPUs")
        except RuntimeError as e:
            print(f"⚠️ Memory growth setup failed: {e}")
            
        return True
    
    def check_data_integrity(self, data_path='datasets/nan_raw_biner/'):
        """Check if training data is accessible and valid"""
        print("🔍 Checking data integrity...")
        
        # Check main data directory
        if not os.path.exists(data_path):
            print(f"❌ Data path not found: {data_path}")
            return False
            
        # Check train/validation directories
        train_path = os.path.join(data_path, 'train', 'images')
        valid_path = os.path.join(data_path, 'validation', 'images')
        
        if not os.path.exists(train_path):
            print(f"❌ Training images path not found: {train_path}")
            return False
            
        if not os.path.exists(valid_path):
            print(f"❌ Validation images path not found: {valid_path}")
            return False
            
        # Count images
        train_images = len(glob(os.path.join(train_path, '*')))
        valid_images = len(glob(os.path.join(valid_path, '*')))
        
        print(f"✅ Found {train_images} training images")
        print(f"✅ Found {valid_images} validation images")
        
        # Check distorted images
        distorted_train = len(glob('datasets/nan_distorted/train/*'))
        distorted_valid = len(glob('datasets/nan_distorted/validation/*'))
        
        print(f"✅ Found {distorted_train} distorted training images")
        print(f"✅ Found {distorted_valid} distorted validation images")
        
        return True
    
    def check_charset_files(self):
        """Check if charset files exist and are valid"""
        print("🔍 Checking charset files...")
        
        charset_file = os.path.join(self.root_path, 'Sets/CHAR_LIST')
        if not os.path.exists(charset_file):
            print(f"❌ Charset file not found: {charset_file}")
            return False
            
        try:
            with open(charset_file, 'r', encoding='utf-8') as f:
                charset = [line.strip() for line in f.readlines()]
            print(f"✅ Charset loaded with {len(charset)} characters")
            return True
        except Exception as e:
            print(f"❌ Error reading charset file: {e}")
            return False
    
    def fix_checkpoint_issues(self, epoch=None):
        """Fix common checkpoint loading issues"""
        print("🔧 Fixing checkpoint issues...")
        
        if epoch is None:
            # Find latest checkpoint
            epoch_dirs = glob(os.path.join(self.results_path, "epoch*"))
            if not epoch_dirs:
                print("❌ No checkpoint directories found")
                return False
                
            epochs = [int(d.split('epoch')[-1]) for d in epoch_dirs if d.split('epoch')[-1].isdigit()]
            epoch = max(epochs) if epochs else None
            
        if epoch is None:
            print("❌ No valid epochs found")
            return False
            
        checkpoint_dir = os.path.join(self.results_path, f"epoch{epoch}", "weights")
        
        if not os.path.exists(checkpoint_dir):
            print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
            return False
            
        # Check if all required weight files exist
        required_files = [
            "gan.weights.h5",
            "generator.weights.h5", 
            "discriminator.weights.h5",
            "rcnn.weights.h5"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(checkpoint_dir, file)):
                missing_files.append(file)
                
        if missing_files:
            print(f"❌ Missing checkpoint files: {missing_files}")
            return False
            
        print(f"✅ All checkpoint files found for epoch {epoch}")
        
        # Create metadata if missing
        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            metadata = {
                'epoch': epoch,
                'scenario': self.scenario,
                'created_by_fix': True,
                'timestamp': 'unknown'
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print("✅ Created missing metadata.json")
            
        return True
    
    def analyze_training_history(self):
        """Analyze training history for issues"""
        print("📊 Analyzing training history...")
        
        history_path = os.path.join(self.results_path, "training_history.json")
        if not os.path.exists(history_path):
            print("⚠️ No training history found")
            return
            
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
                
            epochs = history.get('epochs', [])
            if not epochs:
                print("❌ No epoch data in history")
                return
                
            train_g_loss = history.get('train_g_loss', [])
            val_g_loss = history.get('val_g_loss', [])
            
            # Check for exploding gradients
            if train_g_loss and max(train_g_loss) > 1000:
                print("🚨 Possible exploding gradients detected! Max G loss:", max(train_g_loss))
                
            # Check for vanishing gradients
            if train_g_loss and all(loss < 0.001 for loss in train_g_loss[-5:]):
                print("🚨 Possible vanishing gradients detected!")
                
            # Check convergence
            if len(val_g_loss) > 10:
                recent_trend = np.diff(val_g_loss[-10:])
                if all(trend >= 0 for trend in recent_trend):
                    print("⚠️ Validation loss not improving in last 10 epochs")
                    
            print(f"✅ Analyzed {len(epochs)} epochs of training history")
            
        except Exception as e:
            print(f"❌ Error analyzing training history: {e}")
    
    def create_visualization_samples(self, epoch=None, num_samples=5):
        """Create sample visualizations to check model performance"""
        print("🎨 Creating visualization samples...")
        
        try:
            # Import required modules
            from PIL import Image
            import matplotlib.pyplot as plt
            
            # Check if we have sample images
            sample_dir = 'datasets/nan_distorted/validation'
            sample_files = glob(os.path.join(sample_dir, '*'))[:num_samples]
            
            if not sample_files:
                print("❌ No sample images found for visualization")
                return False
                
            # Create output directory
            viz_dir = os.path.join(self.results_path, "diagnostic_visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            for i, sample_file in enumerate(sample_files):
                try:
                    # Load and process image
                    img = Image.open(sample_file)
                    img_resized = img.resize((1024, 128), Image.LANCZOS)
                    img_gray = img_resized.convert('L')
                    
                    # Save processed sample
                    output_path = os.path.join(viz_dir, f"sample_{i}_processed.png")
                    img_gray.save(output_path)
                    
                except Exception as e:
                    print(f"⚠️ Error processing sample {i}: {e}")
                    continue
                    
            print(f"✅ Created {len(sample_files)} visualization samples in {viz_dir}")
            return True
            
        except ImportError as e:
            print(f"❌ Import error for visualization: {e}")
            return False
    
    def fix_memory_issues(self):
        """Apply fixes for common memory issues"""
        print("🔧 Applying memory fixes...")
        
        # Clear any existing sessions
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Set memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU memory growth enabled")
            except RuntimeError:
                print("⚠️ GPU memory growth already initialized")
                
        # Set environment variables for memory optimization
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        
        print("✅ Memory optimization settings applied")
    
    def validate_batch_size(self, batch_size=12, strategy=None):
        """Validate batch size configuration"""
        print(f"🔍 Validating batch size: {batch_size}")
        
        if strategy is None:
            # Check available GPUs
            gpus = tf.config.experimental.list_physical_devices('GPU')
            num_gpus = len(gpus)
        else:
            num_gpus = strategy.num_replicas_in_sync
            
        if num_gpus == 0:
            print("❌ No GPUs available")
            return False
            
        if batch_size % num_gpus != 0:
            suggested = (batch_size // num_gpus + 1) * num_gpus
            print(f"⚠️ Batch size {batch_size} not divisible by {num_gpus} GPUs")
            print(f"   Suggested batch size: {suggested}")
            return False
            
        per_gpu_batch = batch_size // num_gpus
        print(f"✅ Batch size valid: {batch_size} total ({per_gpu_batch} per GPU)")
        return True
    
    def run_comprehensive_check(self, data_path='datasets/nan_raw_biner/', batch_size=12):
        """Run all diagnostic checks"""
        print("🔍 Running comprehensive training diagnostics...")
        print("=" * 60)
        
        issues_found = []
        
        # Check GPU setup
        if not self.check_gpu_setup():
            issues_found.append("GPU setup")
            
        # Check data integrity
        if not self.check_data_integrity(data_path):
            issues_found.append("Data integrity")
            
        # Check charset files
        if not self.check_charset_files():
            issues_found.append("Charset files")
            
        # Check checkpoint issues
        if not self.fix_checkpoint_issues():
            issues_found.append("Checkpoint files")
            
        # Analyze training history
        self.analyze_training_history()
        
        # Validate batch size
        if not self.validate_batch_size(batch_size):
            issues_found.append("Batch size configuration")
            
        # Apply memory fixes
        self.fix_memory_issues()
        
        # Create visualization samples
        self.create_visualization_samples()
        
        print("=" * 60)
        if issues_found:
            print("❌ Issues found in:")
            for issue in issues_found:
                print(f"   - {issue}")
        else:
            print("✅ All checks passed!")
            
        return len(issues_found) == 0

def main():
    parser = argparse.ArgumentParser(description='Fix GAN-HTR Training Issues')
    parser.add_argument('--scenario', type=str, default='S_iam_OP',
                       help='Training scenario name')
    parser.add_argument('--data-path', type=str, default='datasets/nan_raw_biner/',
                       help='Path to training data')
    parser.add_argument('--batch-size', type=int, default=12,
                       help='Batch size to validate')
    parser.add_argument('--fix-checkpoints', action='store_true',
                       help='Fix checkpoint issues only')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Specific epoch to check/fix')
    
    args = parser.parse_args()
    
    fixer = TrainingIssuesFixer(scenario=args.scenario)
    
    if args.fix_checkpoints:
        success = fixer.fix_checkpoint_issues(args.epoch)
        if success:
            print("✅ Checkpoint issues fixed!")
        else:
            print("❌ Failed to fix checkpoint issues")
    else:
        success = fixer.run_comprehensive_check(args.data_path, args.batch_size)
        if success:
            print("\n🎉 All systems ready for training!")
        else:
            print("\n⚠️ Please fix the identified issues before training")

if __name__ == "__main__":
    main()
