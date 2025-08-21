import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import tensorflow as tf
import numpy as np

def test_unet_batchnorm():
    """Test U-Net architecture with BatchNormalization similar to the original code"""
    print("🔍 Testing U-Net with BatchNormalization...")
    
    try:
        # Create input with same dimensions as original code
        inputs = tf.keras.Input(shape=(128, 1024, 1))
        print(f"Input shape: {inputs.shape}")
        
        # Encoder path (similar to original unet function)
        # Block 1
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        bn1 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv1)
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn1)
        bn1 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn1)
        print(f"Block 1 output shape: {pool1.shape}")
        
        # Block 2
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        bn2 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv2)
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn2)
        bn2 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn2)
        print(f"Block 2 output shape: {pool2.shape}")
        
        # Block 3
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        bn3 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv3)
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn3)
        bn3 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn3)
        print(f"Block 3 output shape: {pool3.shape}")
        
        # Block 4
        conv4 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        bn4 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv4)
        conv4 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn4)
        bn4 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv4)
        drop4 = tf.keras.layers.Dropout(0.5)(bn4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
        print(f"Block 4 output shape: {pool4.shape}")
        
        # Bottleneck
        conv5 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        bn5 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv5)
        conv5 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn5)
        bn5 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv5)
        drop5 = tf.keras.layers.Dropout(0.5)(bn5)
        print(f"Bottleneck output shape: {drop5.shape}")
        
        # Decoder path
        # Block 6
        up6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
        bn6 = tf.keras.layers.BatchNormalization(momentum=0.8)(up6)
        merge6 = tf.keras.layers.concatenate([drop4, bn6])
        conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        bn6 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv6)
        conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn6)
        bn6 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv6)
        print(f"Block 6 output shape: {bn6.shape}")
        
        # Block 7
        up7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(tf.keras.layers.UpSampling2D(size=(2, 2))(bn6))
        bn7 = tf.keras.layers.BatchNormalization(momentum=0.8)(up7)
        merge7 = tf.keras.layers.concatenate([conv3, bn7])
        conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        bn7 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv7)
        conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn7)
        bn7 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv7)
        print(f"Block 7 output shape: {bn7.shape}")
        
        # Block 8
        up8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(tf.keras.layers.UpSampling2D(size=(2, 2))(bn7))
        bn8 = tf.keras.layers.BatchNormalization(momentum=0.8)(up8)
        merge8 = tf.keras.layers.concatenate([conv2, bn8])
        conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        bn8 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv8)
        conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn8)
        bn8 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv8)
        print(f"Block 8 output shape: {bn8.shape}")
        
        # Block 9
        up9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(tf.keras.layers.UpSampling2D(size=(2, 2))(bn8))
        bn9 = tf.keras.layers.BatchNormalization(momentum=0.8)(up9)
        merge9 = tf.keras.layers.concatenate([conv1, bn9])
        conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        bn9 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv9)
        conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn9)
        bn9 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv9)
        conv9 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bn9)
        bn9 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv9)
        print(f"Block 9 output shape: {bn9.shape}")
        
        # Final output
        conv10 = tf.keras.layers.Conv2D(1, 1, activation='tanh')(bn9)
        print(f"Final output shape: {conv10.shape}")
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=conv10)
        print("✅ U-Net model created successfully")
        
        # Test with sample data
        sample_data = np.random.random((2, 128, 1024, 1)).astype(np.float32)
        predictions = model.predict(sample_data[:1], verbose=0)
        print(f"Prediction shape: {predictions.shape}")
        
        return False  # No error
        
    except Exception as e:
        print(f"❌ ERROR in U-Net BatchNormalization test: {e}")
        import traceback
        traceback.print_exc()
        return True

def test_optimizer_with_batchnorm():
    """Test optimizer with BatchNormalization layers to check for variable shape issues"""
    print("\n🔍 Testing optimizer with BatchNormalization...")
    
    try:
        # Create a simple model with BatchNormalization
        inputs = tf.keras.Input(shape=(128, 1024, 1))
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        outputs = tf.keras.layers.Conv2D(1, 1, activation='tanh')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with optimizer similar to original code
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
        model.compile(optimizer=optimizer, loss='mse')
        
        # Check trainable variables
        for i, var in enumerate(model.trainable_variables):
            print(f"Variable {i}: {var.name} - shape: {var.shape}")
            # Check for zero dimensions
            if any(dim == 0 for dim in var.shape):
                print(f"❌ ERROR: Variable {i} has zero dimension: {var.shape}")
                return True
        
        print("✅ All variables have valid dimensions")
        return False
        
    except Exception as e:
        print(f"❌ ERROR in optimizer test: {e}")
        return True

def test_distributed_strategy():
    """Test with distributed strategy like in the original code"""
    print("\n🔍 Testing with distributed strategy...")
    
    try:
        # Setup strategy similar to original code
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
        else:
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
            print("Using OneDeviceStrategy")
        
        with strategy.scope():
            # Create model in strategy scope
            inputs = tf.keras.Input(shape=(128, 1024, 1))
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
            outputs = tf.keras.layers.Conv2D(1, 1, activation='tanh')(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Compile with optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
            model.compile(optimizer=optimizer, loss='mse')
            
            # Check variables
            for i, var in enumerate(model.trainable_variables):
                print(f"Variable {i}: {var.name} - shape: {var.shape}")
                if any(dim == 0 for dim in var.shape):
                    print(f"❌ ERROR: Variable {i} has zero dimension: {var.shape}")
                    return True
            
            print("✅ Distributed strategy test passed")
            return False
            
    except Exception as e:
        print(f"❌ ERROR in distributed strategy test: {e}")
        import traceback
        traceback.print_exc()
        return True

if __name__ == "__main__":
    print("🧪 Detailed BatchNormalization Test")
    print("=" * 40)
    
    # Run all tests
    unet_issue = test_unet_batchnorm()
    optimizer_issue = test_optimizer_with_batchnorm()
    distributed_issue = test_distributed_strategy()
    
    print("\n" + "=" * 40)
    if unet_issue or optimizer_issue or distributed_issue:
        print("🚨 PROBLEM DETECTED: BatchNormalization may be causing issues")
    else:
        print("✅ No issues detected with detailed tests")
        print("💡 The problem might be elsewhere in the original code")