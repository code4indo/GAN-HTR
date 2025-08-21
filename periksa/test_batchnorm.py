import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model

def test_batchnorm_issue():
    """Test to check if BatchNormalization causes zero-dimension tensors"""
    print("🔍 Testing BatchNormalization issue...")
    
    # Create a simple model similar to the problematic one
    inputs = Input(shape=(128, 1024, 1))
    
    # First conv layer
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    print(f"Conv1 shape: {conv1.shape}")
    
    # BatchNormalization that might cause issues
    try:
        bn = BatchNormalization(momentum=0.8)(conv1)
        print(f"BatchNorm shape: {bn.shape}")
        
        # Check if any dimension is zero
        if any(dim == 0 for dim in bn.shape):
            print("❌ ERROR: BatchNormalization produced tensor with zero dimension!")
            return True
        else:
            print("✅ BatchNormalization seems to work correctly")
            return False
    except Exception as e:
        print(f"❌ ERROR in BatchNormalization: {e}")
        return True

def test_model_creation():
    """Test full model creation to check for issues"""
    print("\n🔍 Testing full model creation...")
    
    try:
        inputs = Input(shape=(128, 1024, 1))
        
        # First block
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        bn1 = BatchNormalization(momentum=0.8)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
        print(f"Block 1 output shape: {pool1.shape}")
        
        # Second block
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        bn2 = BatchNormalization(momentum=0.8)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
        print(f"Block 2 output shape: {pool2.shape}")
        
        # Third block
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        bn3 = BatchNormalization(momentum=0.8)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
        print(f"Block 3 output shape: {pool3.shape}")
        
        print("✅ Model creation successful - no zero dimensions detected")
        return False
        
    except Exception as e:
        print(f"❌ ERROR in model creation: {e}")
        return True

def test_with_sample_data():
    """Test with actual sample data"""
    print("\n🔍 Testing with sample data...")
    
    try:
        # Create sample data with the same dimensions as used in the main code
        sample_data = np.random.random((2, 128, 1024, 1)).astype(np.float32)
        print(f"Sample data shape: {sample_data.shape}")
        
        # Create a simple model
        inputs = Input(shape=(128, 1024, 1))
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        bn = BatchNormalization(momentum=0.8)(conv1)
        outputs = Conv2D(1, 1, activation='tanh')(bn)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        
        # Test forward pass
        predictions = model.predict(sample_data[:1], verbose=0)
        print(f"Prediction shape: {predictions.shape}")
        
        # Check for zero dimensions
        if any(dim == 0 for dim in predictions.shape):
            print("❌ ERROR: Model produced output with zero dimension!")
            return True
        else:
            print("✅ Model works correctly with sample data")
            return False
            
    except Exception as e:
        print(f"❌ ERROR in sample data test: {e}")
        return True

if __name__ == "__main__":
    print("🧪 BatchNormalization Dimension Test")
    print("=" * 40)
    
    # Run all tests
    bn_issue = test_batchnorm_issue()
    model_issue = test_model_creation()
    data_issue = test_with_sample_data()
    
    print("\n" + "=" * 40)
    if bn_issue or model_issue or data_issue:
        print("🚨 PROBLEM DETECTED: BatchNormalization may be causing zero-dimension tensors")
    else:
        print("✅ No issues detected with current test setup")