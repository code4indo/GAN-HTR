"""
Fix untuk error DistributedVariable.handle outside replica context

Error terjadi karena:
1. Inisialisasi optimizer dilakukan di luar strategy.scope()
2. Gradient computation mencoba mengakses DistributedVariable di luar replica context
3. Perlu memindahkan semua operasi yang melibatkan distributed variables ke dalam strategy.run()

Solusi:
1. Pindahkan inisialisasi optimizer ke dalam strategy.run()
2. Gunakan dummy forward pass yang aman untuk distributed training
3. Pastikan semua gradient operations berada dalam strategy context
"""

import tensorflow as tf

def fix_optimizer_initialization_section():
    """
    Perbaikan untuk bagian inisialisasi optimizer yang menyebabkan error
    """
    
    fixed_code = '''
    # Create optimizers and define training step in strategy scope
    with strategy.scope():
        print("🔧 Creating optimizers in distributed strategy scope...")
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr, beta_1=0.5)
        disc1_optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr, beta_1=0.5)
        disc2_optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr, beta_1=0.5)
        
        # Inisialisasi optimizer variables dengan dummy gradients dalam strategy context
        print("🔧 Initializing optimizer variables with dummy gradients...")
        
        @tf.function
        def initialize_optimizers():
            """Initialize all optimizers safely within distributed strategy"""
            def init_step():
                # Create dummy inputs with proper batch size per replica
                per_replica_batch = batch_size // strategy.num_replicas_in_sync
                sample_input = tf.random.normal((per_replica_batch, 128, 1024, 1))
                sample_target = tf.random.normal((per_replica_batch, 128, 1024, 1))
                sample_crnn_input = tf.random.normal((per_replica_batch, 1024, 128, 1))
                
                # Generator initialization
                with tf.GradientTape() as tape:
                    fake_output = generator(sample_input, training=True)
                    fake_output = tf.cast(fake_output, tf.float32)
                    sample_target = tf.cast(sample_target, tf.float32)
                    fake_loss = tf.reduce_mean(tf.square(fake_output - sample_target))
                
                gen_grads = tape.gradient(fake_loss, generator.trainable_variables)
                gen_grads_filtered = [grad for grad in gen_grads if grad is not None]
                gen_vars_filtered = [var for var, grad in zip(generator.trainable_variables, gen_grads) if grad is not None]
                if gen_grads_filtered:
                    gen_optimizer.apply_gradients(zip(gen_grads_filtered, gen_vars_filtered))
                
                # Discriminator 1 initialization
                with tf.GradientTape() as tape:
                    real_pred = discriminator_1([sample_target, sample_input], training=True)
                    fake_pred = discriminator_1([fake_output, sample_input], training=True)
                    real_pred = tf.cast(real_pred, tf.float32)
                    fake_pred = tf.cast(fake_pred, tf.float32)
                    d1_loss = tf.reduce_mean(tf.square(real_pred - 1.0)) + tf.reduce_mean(tf.square(fake_pred))
                
                d1_grads = tape.gradient(d1_loss, discriminator_1.trainable_variables)
                d1_grads_filtered = [grad for grad in d1_grads if grad is not None]
                d1_vars_filtered = [var for var, grad in zip(discriminator_1.trainable_variables, d1_grads) if grad is not None]
                if d1_grads_filtered:
                    disc1_optimizer.apply_gradients(zip(d1_grads_filtered, d1_vars_filtered))
                
                # Discriminator 2 initialization
                with tf.GradientTape() as tape:
                    crnn_pred = discriminator_2(sample_crnn_input, training=True)
                    crnn_pred = tf.cast(crnn_pred, tf.float32)
                    d2_loss = tf.reduce_mean(tf.square(crnn_pred))
                
                d2_grads = tape.gradient(d2_loss, discriminator_2.trainable_variables)
                d2_grads_filtered = [grad for grad in d2_grads if grad is not None]
                d2_vars_filtered = [var for var, grad in zip(discriminator_2.trainable_variables, d2_grads) if grad is not None]
                if d2_grads_filtered:
                    disc2_optimizer.apply_gradients(zip(d2_grads_filtered, d2_vars_filtered))
                
                return tf.constant(0.0)  # Return dummy value
            
            # Run initialization in distributed context
            strategy.run(init_step)
        
        # Execute initialization
        initialize_optimizers()
        print("✅ All optimizer variables initialized successfully!")
    '''
    
    return fixed_code

if __name__ == "__main__":
    print("Fix untuk error DistributedVariable.handle outside replica context")
    print("\nKode yang diperbaiki:")
    print(fix_optimizer_initialization_section())
