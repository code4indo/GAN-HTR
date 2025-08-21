import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import tensorflow as tf
import numpy as np

def test_optimizer_variable_shapes():
    """Test to check if optimizer variables have zero dimensions"""
    print("🔍 Testing optimizer variable shapes...")
    
    try:
        # Create a simple model similar to the problematic one
        inputs = tf.keras.Input(shape=(128, 1024, 1))
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        bn = tf.keras.layers.BatchNormalization(momentum=0.8)(conv1)
        outputs = tf.keras.layers.Conv2D(1, 1, activation='tanh')(bn)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with optimizer similar to original code
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
        model.compile(optimizer=optimizer, loss='mse')
        
        # Create sample data
        sample_data = np.random.random((2, 128, 1024, 1)).astype(np.float32)
        sample_targets = np.random.random((2, 128, 1024, 1)).astype(np.float32)
        
        # Run a training step to initialize optimizer variables
        print("Running a training step to initialize optimizer variables...")
        model.train_on_batch(sample_data[:1], sample_targets[:1])
        
        # Check model variables
        print("\nModel variables:")
        for i, var in enumerate(model.trainable_variables):
            print(f"  {i}: {var.name} - shape: {var.shape}")
            if any(dim == 0 for dim in var.shape):
                print(f"    ❌ ZERO DIMENSION DETECTED!")
                return True
        
        # Check optimizer variables
        print("\nOptimizer variables:")
        for i, var in enumerate(optimizer.variables):
            print(f"  {i}: {var.name} - shape: {var.shape}")
            if any(dim == 0 for dim in var.shape):
                print(f"    ❌ ZERO DIMENSION DETECTED!")
                return True
                
        print("✅ All variables have valid dimensions")
        return False
        
    except Exception as e:
        print(f"❌ ERROR in optimizer variable test: {e}")
        import traceback
        traceback.print_exc()
        return True

def test_specific_adam_optimizer_issue():
    """Test specifically for the Adam optimizer issue mentioned in the error"""
    print("\n🔍 Testing specific Adam optimizer issue...")
    
    try:
        # Create a model with BatchNormalization that might cause issues
        inputs = tf.keras.Input(shape=(128, 1024, 1))
        
        # This is similar to the structure in the original unet function
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        bn1 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv1)
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn1)
        bn1 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv1)
        
        outputs = tf.keras.layers.Conv2D(1, 1, activation='tanh')(bn1)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Use the exact same optimizer configuration as in the error
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
        model.compile(optimizer=optimizer, loss='mse')
        
        # Create sample data
        sample_data = np.random.random((2, 128, 1024, 1)).astype(np.float32)
        sample_targets = np.random.random((2, 128, 1024, 1)).astype(np.float32)
        
        # Try to train and see if we get the specific error
        print("Attempting training step...")
        loss = model.train_on_batch(sample_data[:1], sample_targets[:1])
        print(f"Training step completed successfully, loss: {loss}")
        
        # Check for any variables with zero dimensions
        print("\nChecking for zero-dimension variables...")
        for var in model.trainable_variables:
            if any(dim == 0 for dim in var.shape):
                print(f"❌ Model variable with zero dimension: {var.name} - {var.shape}")
                return True
                
        for var in optimizer.variables:
            if any(dim == 0 for dim in var.shape):
                print(f"❌ Optimizer variable with zero dimension: {var.name} - {var.shape}")
                return True
        
        print("✅ No zero-dimension variables found")
        return False
        
    except Exception as e:
        print(f"❌ ERROR in specific Adam test: {e}")
        import traceback
        traceback.print_exc()
        return True

def test_gradient_computation():
    """Test gradient computation which is where the error occurs"""
    print("\n🔍 Testing gradient computation...")
    
    try:
        # Create the same model structure
        inputs = tf.keras.Input(shape=(128, 1024, 1))
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        bn1 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv1)
        outputs = tf.keras.layers.Conv2D(1, 1, activation='tanh')(bn1)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
        
        # Create sample data
        sample_data = tf.constant(np.random.random((1, 128, 1024, 1)).astype(np.float32))
        sample_targets = tf.constant(np.random.random((1, 128, 1024, 1)).astype(np.float32))
        
        # Compute gradients - this is where the error typically occurs
        with tf.GradientTape() as tape:
            predictions = model(sample_data, training=True)
            loss = tf.keras.losses.mse(sample_targets, predictions)
            loss = tf.reduce_mean(loss)
        
        print(f"Loss computed successfully: {loss}")
        
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        print(f"Gradients computed for {len(gradients)} variables")
        
        # Check gradients for None values or zero shapes
        for i, (grad, var) in enumerate(zip(gradients, model.trainable_variables)):
            if grad is None:
                print(f"❌ Gradient {i} is None for variable {var.name}")
                return True
            print(f"  Gradient {i}: {var.name} - grad shape: {grad.shape}, var shape: {var.shape}")
            
            # Check for zero dimensions in gradients
            if any(dim == 0 for dim in grad.shape):
                print(f"❌ Gradient {i} has zero dimension: {grad.shape}")
                return True
                
        print("✅ Gradients computed successfully with no zero dimensions")
        return False
        
    except Exception as e:
        print(f"❌ ERROR in gradient computation test: {e}")
        import traceback
        traceback.print_exc()
        return True

if __name__ == "__main__":
    print("🧪 Optimizer Variable Shape Test")
    print("=" * 40)
    
    # Run all tests
    var_issue = test_optimizer_variable_shapes()
    adam_issue = test_specific_adam_optimizer_issue()
    grad_issue = test_gradient_computation()
    
    print("\n" + "=" * 40)
    if var_issue or adam_issue or grad_issue:
        print("🚨 PROBLEM DETECTED: Optimizer variables may have zero dimensions")
        print("💡 This could be the source of the original error")
    else:
        print("✅ No optimizer variable issues detected")
        print("💡 The problem might be in the distributed training setup or specific to the original code")