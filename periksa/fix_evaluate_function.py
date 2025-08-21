#!/usr/bin/env python3
"""
Script untuk memperbaiki fungsi evaluate di jnm_GAN_AHTR.py
Menambahkan pemanggilan get_psnr_iam() dalam fungsi evaluate
"""

import sys
import os
sys.path.append('/home/lambda_one/tesis/GAN-HTR')

def fix_evaluate_function():
    """
    Menambahkan pemanggilan get_psnr_iam() ke dalam fungsi evaluate
    """
    
    file_path = '/home/lambda_one/tesis/GAN-HTR/jnm_GAN_AHTR.py'
    
    # Baca file asli
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Cari lokasi akhir fungsi evaluate
    # Fungsi evaluate berakhir sebelum def loadCRNNModel
    evaluate_end_marker = "def loadCRNNModel(epoch,mode_crnn='no_progressive', batch_size=12):"
    
    if evaluate_end_marker in content:
        # Split konten di marker
        parts = content.split(evaluate_end_marker)
        
        if len(parts) == 2:
            # Tambahkan pemanggilan get_psnr_iam di akhir fungsi evaluate
            psnr_call_code = """
	# Calculate PSNR metrics after visual evaluation
	print(f"📊 Calculating PSNR metrics for epoch {epoch}...")
	try:
		average_psnr = get_psnr_iam()
		if average_psnr:
			print(f"✅ Epoch {epoch} - Average PSNR: {average_psnr:.2f} dB")
		else:
			print(f"⚠️ Epoch {epoch} - PSNR calculation failed")
	except Exception as e:
		print(f"❌ Epoch {epoch} - PSNR calculation error: {e}")

"""
            
            # Gabungkan kembali
            new_content = parts[0] + psnr_call_code + evaluate_end_marker + parts[1]
            
            # Tulis file baru
            backup_path = file_path + '.backup'
            
            # Buat backup
            with open(backup_path, 'w') as f:
                f.write(content)
            print(f"✅ Backup created: {backup_path}")
            
            # Tulis file yang diperbaiki
            with open(file_path, 'w') as f:
                f.write(new_content)
            
            print(f"✅ Fixed evaluate function in {file_path}")
            print("📊 Now the evaluate function will call get_psnr_iam() and display PSNR metrics!")
            
            return True
        else:
            print("❌ Error: Multiple or no occurrences of loadCRNNModel marker found")
            return False
    else:
        print("❌ Error: loadCRNNModel marker not found in file")
        return False

def test_quick_training():
    """
    Test training dengan 1 epoch untuk memverifikasi PSNR ditampilkan
    """
    print("\n🚀 Testing quick training to verify PSNR display...")
    
    # Import setelah fix
    try:
        from jnm_GAN_AHTR import train_GAN_crnn
        
        print("📋 Running 1 epoch training to test PSNR display...")
        print("⏱️ This should take 2-5 minutes...")
        
        # Run 1 epoch dengan batch size kecil
        train_GAN_crnn(nepochs=1, batch_size=4)
        
        print("✅ Training completed! Check the output above for PSNR metrics.")
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")

if __name__ == '__main__':
    print("🔧 Fixing evaluate function to display PSNR metrics...")
    
    success = fix_evaluate_function()
    
    if success:
        print("\n" + "="*60)
        print("✅ FIX APPLIED SUCCESSFULLY!")
        print("📊 Now when you run training, PSNR will be displayed!")
        print("="*60)
        
        # Tanya user apakah ingin test langsung
        response = input("\n🚀 Do you want to test with 1 epoch training now? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            test_quick_training()
        else:
            print("\n📋 To test manually, run:")
            print("poetry run python jnm_GAN_AHTR.py --epochs 1 --batch-size 4")
            print("\n📊 You should now see PSNR metrics during evaluation!")
    else:
        print("\n❌ Fix failed. Please check the error messages above.")