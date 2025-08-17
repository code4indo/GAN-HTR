
# SCRIPT TEST INDIVIDUAL COMPONENTS
# Untuk mengisolasi masalah pada setiap bagian

def test_data_loading():
    """Test data loading tanpa training"""
    print("🧪 Testing data loading...")
    
    # Test basic data loading
    list_image_train = read_file_shuffle(rootPath + 'Sets/list_train_nan.txt')[:5]
    list_lines = read_file(rootPath + 'Sets/lines.txt')
    
    for im_base in list_image_train:
        try:
            # Test image loading
            deg_image, gt_image = readGrayPair(im_base + '.png', split='train')
            print(f"✅ Image {im_base}: {deg_image.shape}, {gt_image.shape}")
            
            # Check for NaN values
            if np.any(np.isnan(deg_image)) or np.any(np.isnan(gt_image)):
                print(f"❌ NaN detected in {im_base}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading {im_base}: {e}")
            return False
    
    print("✅ Data loading test passed")
    return True

def test_model_creation():
    """Test model creation tanpa training"""
    print("🧪 Testing model creation...")
    
    try:
        generator = unet()
        discriminator_1 = build_discriminator_1()
        discriminator_2 = build_discriminator_2()
        
        print("✅ All models created successfully")
        
        # Test forward pass
        dummy_input = np.random.random((1, 128, 1024, 1)).astype(np.float32)
        gen_output = generator(dummy_input)
        print(f"✅ Generator output shape: {gen_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_ctc_loss():
    """Test CTC loss dengan sample data"""
    print("🧪 Testing CTC loss...")
    
    # Sample data
    batch_size = 2
    seq_length = 256
    vocab_size = len(charset_base) + 1
    max_label_length = 10
    
    # Create sample predictions dan labels
    y_pred = tf.random.normal((batch_size, seq_length, vocab_size))
    y_true = tf.random.uniform((batch_size, max_label_length), 
                              maxval=vocab_size-1, dtype=tf.int32)
    
    try:
        loss = improved_ctc_loss_lambda_func(y_true, y_pred)
        print(f"✅ CTC loss computed: {float(loss):.4f}")
        
        if tf.math.is_finite(loss):
            print("✅ CTC loss is finite")
            return True
        else:
            print("❌ CTC loss is not finite")
            return False
            
    except Exception as e:
        print(f"❌ CTC loss failed: {e}")
        return False

# RUN ALL TESTS
if __name__ == "__main__":
    print("🚀 RUNNING COMPONENT TESTS")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation), 
        ("CTC Loss", test_ctc_loss)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    print(f"\n🏁 TEST RESULTS:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    if all(results.values()):
        print("\n🎉 All tests passed! Ready for conservative training.")
    else:
        print("\n🚨 Some tests failed. Fix issues before training.")
