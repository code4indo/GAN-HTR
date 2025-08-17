"""
Solusi untuk mengatasi error training GAN-HTR

Masalah yang ditemukan:
1. Iterator Incarnation Error dalam multi-GPU distributed training
2. JSON serialization error saat menyimpan metadata dengan exception object
3. Memory leakage dan register spilling warnings

Solusi yang akan diterapkan:
1. Perbaiki distributed dataset handling
2. Perbaiki error handling dalam fungsi save
3. Optimasi memory management untuk mengurangi register spilling
"""

import tensorflow as tf
import json
import os
import time

def fix_distributed_dataset_iterator():
    """
    Perbaikan untuk masalah Invalid incarnation id dalam distributed training
    """
    solution_code = '''
def create_optimized_dataset_fixed(list_images, list_lines, mode, strategy, batch_size=4):
    """
    Dataset creation with improved distributed training support
    """
    def generator():
        indices = list(range(len(list_images)))
        if mode == 'train':
            import random
            random.shuffle(indices)
        
        for idx in indices:
            try:
                image_path = list_images[idx]
                line = list_lines[idx]
                
                # Load and preprocess image
                image = tf.io.read_file(image_path)
                image = tf.image.decode_image(image, channels=3)
                image = tf.cast(image, tf.float32) / 255.0
                
                # Ensure consistent shape
                image = tf.image.resize(image, [64, 256])
                
                yield image, line
                
            except Exception as e:
                print(f"Warning: Skipping corrupt image {image_path}: {e}")
                continue
    
    # Create dataset with proper configuration for distributed training
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=[64, 256, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.string)
        )
    )
    
    # Configure for distributed training
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Important: Use experimental_distribute_dataset properly
    return dataset

def fix_iterator_reset():
    """
    Perbaikan untuk reset iterator antar epoch
    """
    solution_code = '''
    # Dalam training loop, pastikan iterator direset dengan benar
    for e in range(ep_start, epochs):
        print(f"🔄 Epoch {e}/{epochs} | LR: {optimizer.learning_rate:.6f}")
        
        try:
            # Recreate dataset untuk setiap epoch untuk menghindari incarnation error
            dataset_train = create_optimized_dataset_fixed(
                list_image_train, list_lines, 'train', strategy, batch_size
            )
            # PENTING: Distribute dataset baru untuk setiap epoch
            distributed_dataset_train = strategy.experimental_distribute_dataset(dataset_train)
            
            # Training loop
            for batch_data in distributed_dataset_train:
                # ... training code ...
                pass
                
        except tf.errors.InvalidArgumentError as e:
            if "Invalid incarnation id" in str(e):
                print("🔄 Iterator incarnation error detected, recreating dataset...")
                # Recreate dataset dan coba lagi
                dataset_train = create_optimized_dataset_fixed(
                    list_image_train, list_lines, 'train', strategy, batch_size
                )
                distributed_dataset_train = strategy.experimental_distribute_dataset(dataset_train)
                continue
            else:
                raise e
    '''
    
    return solution_code
'''

def fix_json_serialization_error():
    """
    Perbaikan untuk error JSON serialization
    """
    solution_code = '''
def save_fixed(gan, generator, discriminator_1, discriminator_2, epoch, error_info=None):
    """Enhanced save function with proper error handling"""
    try:
        # Create directory if it doesn't exist
        save_dir = os.path.join(rootPath, "ResultGan" + scenario, "epoch" + str(epoch), "weights")
        os.makedirs(save_dir, exist_ok=True)

        # Save models
        print(f"💾 Saving models for epoch {epoch}...")
        
        gan.save_weights(os.path.join(save_dir, "gan.weights.h5"))
        print("   ✅ GAN weights saved")
        
        discriminator_1.save_weights(os.path.join(save_dir, "discriminator.weights.h5"))
        print("   ✅ Discriminator 1 weights saved")
        
        discriminator_2.save_weights(os.path.join(save_dir, "rcnn.weights.h5"))
        print("   ✅ RCNN weights saved")
        
        generator.save_weights(os.path.join(save_dir, "generator.weights.h5"))
        print("   ✅ Generator weights saved")
        
        # Save training metadata dengan error handling yang benar
        metadata = {
            'epoch': epoch,
            'scenario': scenario,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'emergency_save' if error_info else 'normal_save'
        }
        
        # PENTING: Jangan serialize exception objects!
        if error_info:
            # Konversi exception ke string yang bisa di-serialize
            metadata['error_message'] = str(error_info)
            metadata['error_type'] = type(error_info).__name__
        
        import json
        with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📁 All models saved successfully to: {save_dir}")
        
    except Exception as save_error:
        print(f"❌ Error saving models: {save_error}")
        # Log error tapi jangan raise jika ini emergency save
        if error_info is None:  # Only raise if this wasn't already an emergency save
            raise save_error
'''
    
    return solution_code

def fix_memory_optimization():
    """
    Optimasi memory untuk mengurangi register spilling
    """
    solution_code = '''
# Konfigurasi GPU memory growth untuk mengurangi memory pressure
def configure_gpu_memory():
    """Configure GPU memory to reduce register spilling"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Enable memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set virtual GPU configuration untuk memory management yang lebih baik
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]  # 8GB limit
                )
            print(f"✅ Configured {len(gpus)} GPUs with memory growth")
        except Exception as e:
            print(f"⚠️ GPU configuration warning: {e}")

# Optimasi batch size berdasarkan available memory
def get_optimal_batch_size():
    """Calculate optimal batch size based on available GPU memory"""
    try:
        gpu_info = tf.config.experimental.get_memory_info('GPU:0')
        available_memory = gpu_info['current'] / (1024**3)  # GB
        
        if available_memory > 12:
            return 8
        elif available_memory > 8:
            return 6
        elif available_memory > 6:
            return 4
        else:
            return 2
    except:
        return 4  # Default fallback

# Mixed precision untuk mengurangi memory usage
def enable_mixed_precision():
    """Enable mixed precision training to reduce memory usage"""
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("✅ Mixed precision enabled")
'''
    
    return solution_code

if __name__ == "__main__":
    print("🔧 Panduan Perbaikan Error Training GAN-HTR")
    print("=" * 60)
    
    print("\n1. MASALAH ITERATOR INCARNATION ERROR:")
    print("   - Disebabkan oleh distributed dataset yang tidak direset dengan benar")
    print("   - Solusi: Recreate dataset untuk setiap epoch")
    
    print("\n2. MASALAH JSON SERIALIZATION ERROR:")
    print("   - Exception objects tidak bisa diserialisasi ke JSON")
    print("   - Solusi: Konversi exception ke string sebelum save")
    
    print("\n3. MASALAH REGISTER SPILLING:")
    print("   - Memory pressure yang tinggi menyebabkan register spill")
    print("   - Solusi: Enable memory growth dan mixed precision")
    
    print("\n4. RECOMMENDED ACTIONS:")
    print("   - Turunkan batch size dari 8 ke 4 atau 6")
    print("   - Enable GPU memory growth")
    print("   - Tambahkan error handling yang lebih robust")
    print("   - Gunakan mixed precision training")
