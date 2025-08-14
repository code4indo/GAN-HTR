"""
Hardware Analysis dan Optimasi untuk Training GAN-HTR
======================================================

Spesifikasi Hardware Sistem:
"""

import os
import psutil

class HardwareAnalysis:
    def __init__(self):
        self.specs = {
            "CPU": {
                "model": "AMD Ryzen Threadripper PRO 3955WX",
                "cores": 16,
                "threads": 32,
                "base_freq": "2.2 GHz",
                "boost_freq": "3.9 GHz",
                "cache_l3": "64 MB",
                "architecture": "Zen 2",
                "rating": "SANGAT BAGUS"
            },
            "GPU": {
                "count": 2,
                "model": "NVIDIA RTX A4000",
                "memory_per_gpu": "16 GB GDDR6",
                "total_gpu_memory": "32 GB",
                "compute_capability": "8.6",
                "cuda_cores": 6144,
                "tensor_cores": "2nd Gen RT Cores",
                "rating": "SANGAT BAGUS"
            },
            "RAM": {
                "total": "125 GB",
                "available": "104 GB",
                "type": "DDR4",
                "rating": "SANGAT BAGUS"
            },
            "Storage": {
                "type": "NVMe SSD",
                "size": "1.8 TB",
                "free": "318 GB",
                "rating": "SANGAT BAGUS"
            }
        }
    
    def print_analysis(self):
        print("="*60)
        print("ANALISIS HARDWARE UNTUK TRAINING GAN-HTR")
        print("="*60)
        
        for component, details in self.specs.items():
            print(f"\n{component}:")
            for key, value in details.items():
                if key != "rating":
                    print(f"  {key}: {value}")
            print(f"  Status: {details['rating']}")
        
        print("\n" + "="*60)
        print("REKOMENDASI OPTIMASI")
        print("="*60)
        
        optimizations = self.get_optimizations()
        for category, opts in optimizations.items():
            print(f"\n{category.upper()}:")
            for opt in opts:
                print(f"  • {opt}")

    def get_optimizations(self):
        return {
            "immediate_improvements": [
                "Gunakan kedua GPU secara bersamaan (Multi-GPU training)",
                "Tingkatkan batch size menjadi 4-6 per GPU (total 8-12)",
                "Aktifkan data parallelism dengan tf.distribute.Strategy",
                "Optimalkan data loading dengan tf.data pipeline",
                "Gunakan gradient accumulation untuk batch size yang lebih besar"
            ],
            
            "memory_optimizations": [
                "Aktifkan AMP (Automatic Mixed Precision) - sudah dilakukan",
                "Gunakan gradient checkpointing untuk model besar",
                "Implementasi memory-efficient attention mechanisms",
                "Optimalkan buffer sizes untuk dataset",
                "Set GPU memory growth limit yang optimal"
            ],
            
            "cpu_optimizations": [
                "Gunakan semua 32 threads untuk data preprocessing",
                "Implementasi multiprocessing untuk data augmentation",
                "Optimalkan tf.data.Dataset dengan parallel processing",
                "Gunakan CPU untuk data loading sambil GPU training",
                "Set optimal number of parallel calls"
            ],
            
            "advanced_optimizations": [
                "Implementasi Model Parallelism untuk model yang sangat besar",
                "Gunakan XLA (Accelerated Linear Algebra) compilation",
                "Optimalkan model architecture untuk efisiensi",
                "Implementasi custom training loops untuk kontrol penuh",
                "Gunakan TensorBoard profiler untuk identifikasi bottleneck"
            ]
        }

class MultiGPUOptimizer:
    """Implementasi optimasi Multi-GPU untuk GAN-HTR"""
    
    @staticmethod
    def create_distributed_strategy():
        """Setup strategi distribusi untuk multi-GPU"""
        import tensorflow as tf
        
        # Deteksi GPU yang tersedia
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(f"Detected {len(gpus)} GPUs")
        
        if len(gpus) > 1:
            # Mirror Strategy untuk multi-GPU single-node
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
            return strategy
        else:
            print("Using single GPU strategy")
            return tf.distribute.get_strategy()
    
    @staticmethod
    def optimize_data_pipeline(dataset, batch_size_per_replica):
        """Optimasi pipeline data untuk multi-GPU"""
        import tensorflow as tf
        
        # Auto-tune untuk performance optimal
        AUTOTUNE = tf.data.AUTOTUNE
        
        # Optimalkan pipeline
        dataset = dataset.cache()  # Cache dataset di memory
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size_per_replica * 2)  # Batch untuk 2 GPU
        dataset = dataset.prefetch(AUTOTUNE)
        
        return dataset
    
    @staticmethod
    def get_optimal_batch_sizes():
        """Menentukan batch size optimal berdasarkan hardware"""
        return {
            "single_gpu": 4,  # Conservative untuk mencegah OOM
            "dual_gpu": 8,    # 4 per GPU
            "aggressive": 12  # 6 per GPU jika memori cukup
        }

def create_optimization_script():
    """Generate script optimasi untuk implementasi"""
    script_content = '''
import tensorflow as tf
import gc

# Konfigurasi Multi-GPU
def setup_multi_gpu():
    """Setup konfigurasi multi-GPU optimal"""
    
    # Deteksi dan konfigurasi GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth untuk semua GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set logical devices jika perlu
            # tf.config.experimental.set_logical_device_configuration(
            #     gpus[0], 
            #     [tf.config.experimental.LogicalDeviceConfiguration(memory_limit=14000)]
            # )
            
            print(f"Configured {len(gpus)} GPUs with memory growth")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Setup distributed strategy
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy across {strategy.num_replicas_in_sync} GPUs")
    else:
        strategy = tf.distribute.get_strategy()
        print("Using single device strategy")
    
    return strategy

# Optimasi Data Pipeline
def create_optimized_dataset(data_generator, batch_size_per_replica):
    """Buat dataset yang dioptimasi untuk multi-GPU"""
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Buat dataset dari generator
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature={
            'deg_image': tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
            'gt_image': tf.TensorSpec(shape=(128, 1024, 1), dtype=tf.float32),
            'crnn_image': tf.TensorSpec(shape=(1024, 128, 1), dtype=tf.float32),
            'transcription': tf.TensorSpec(shape=(max_text_length,), dtype=tf.int16),
            'text_line': tf.TensorSpec(shape=(), dtype=tf.string)
        }
    )
    
    # Pipeline optimizations
    dataset = dataset.cache()  # Cache di memory
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size_per_replica, drop_remainder=True)
    dataset = dataset.repeat()  # Infinite dataset
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

# Training Loop yang Dioptimasi
def optimized_training_step(models, batch_data, strategy):
    """Training step yang dioptimasi untuk multi-GPU"""
    
    def train_step(inputs):
        batch_train, batch_target, x_train_rcnn, y_train_rcnn = inputs
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Forward pass
            generated_images = models['generator'](batch_train, training=True)
            
            # Discriminator predictions
            real_output = models['discriminator'](batch_target, training=True)
            fake_output = models['discriminator'](generated_images, training=True)
            
            # Losses calculation
            gen_loss = generator_loss(fake_output, batch_target, generated_images)
            disc_loss = discriminator_loss(real_output, fake_output)
        
        # Gradients calculation
        gen_gradients = gen_tape.gradient(gen_loss, models['generator'].trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, models['discriminator'].trainable_variables)
        
        # Apply gradients
        models['gen_optimizer'].apply_gradients(zip(gen_gradients, models['generator'].trainable_variables))
        models['disc_optimizer'].apply_gradients(zip(disc_gradients, models['discriminator'].trainable_variables))
        
        return gen_loss, disc_loss
    
    # Distributed training step
    per_replica_losses = strategy.run(train_step, args=(batch_data,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# Memory Management
def advanced_memory_management():
    """Manajemen memori yang lebih agresif"""
    
    # Clear unnecessary variables
    gc.collect()
    
    # Tensorflow memory cleanup
    tf.keras.backend.clear_session()
    
    # Force GPU memory cleanup
    if tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.reset_memory_stats('GPU:0')
        if len(tf.config.experimental.list_physical_devices('GPU')) > 1:
            tf.config.experimental.reset_memory_stats('GPU:1')

'''
    
    return script_content

if __name__ == "__main__":
    # Analisis hardware
    analyzer = HardwareAnalysis()
    analyzer.print_analysis()
    
    print("\n" + "="*60)
    print("ESTIMASI PERFORMA IMPROVEMENT")
    print("="*60)
    
    improvements = {
        "Current (Single GPU, batch=2)": "Baseline",
        "Dual GPU (batch=4 each)": "4x speedup teoritis",
        "Optimized pipeline": "+20-30% improvement",
        "XLA compilation": "+10-15% improvement", 
        "Total potential": "5-6x faster training"
    }
    
    for optimization, benefit in improvements.items():
        print(f"  {optimization}: {benefit}")
    
    print(f"\n{'='*60}")
    print("KESIMPULAN: Hardware Anda SANGAT POWERFUL!")
    print("Dengan optimasi yang tepat, training bisa 5-6x lebih cepat.")
    print(f"{'='*60}")
