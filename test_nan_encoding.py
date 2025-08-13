#!/usr/bin/env python3
"""
Simple test untuk validasi charset dengan menggunakan fungsi encode_txt
"""

import os
import sys

def read_file_char(file_path):
    """Read character list from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def normalizeTranscription(text):
    """Normalize transcription text (simplified version)"""
    # Convert to lowercase and clean up
    text = text.lower()
    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text

def encode_txt(text, charset_base):
    """Encode text using charset (from jnm_GAN_AHTR.py)"""
    encoded = []
    cc = text.split()
    for item in cc:
        try:
            index = charset_base.index(item)
            encoded.append(index)
        except ValueError:
            print(f"ERROR: Word '{item}' not found in charset_base")
            # Use <UNK> token
            unk_index = charset_base.index('<UNK>')
            encoded.append(unk_index)
    return encoded

def test_nan_encoding():
    """Test encoding dengan format NaN dataset"""
    
    # Load charset
    charset_base = read_file_char('Sets/CHAR_LIST')
    print(f"Loaded charset with {len(charset_base)} tokens")
    
    # Test dengan sample dari dataset NaN
    lines_file = 'datasets/nan_raw_biner/train/lines.txt'
    
    print("\n=== Testing NaN Dataset Encoding ===")
    
    with open(lines_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num >= 5:  # Test first 5 lines only
                break
                
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                if len(parts) >= 2:
                    filename = parts[0]
                    text_line = parts[1]
                    
                    print(f"\nTest {line_num + 1}: {filename}")
                    print(f"Original text: {text_line}")
                    
                    # Normalize
                    normalized = normalizeTranscription(text_line)
                    print(f"Normalized: {normalized}")
                    
                    # Encode
                    try:
                        encoded = encode_txt(normalized, charset_base)
                        print(f"✅ Encoded successfully: length={len(encoded)}")
                        print(f"   First 5 indices: {encoded[:5]}")
                        
                        # Verify by decoding first few words
                        decoded_words = []
                        for idx in encoded[:5]:
                            if idx < len(charset_base):
                                decoded_words.append(charset_base[idx])
                        print(f"   Decoded first 5: {' '.join(decoded_words)}")
                        
                    except Exception as e:
                        print(f"❌ Encoding failed: {e}")

def main():
    print("Testing NaN Dataset Encoding...")
    test_nan_encoding()
    print("\n✅ NaN dataset encoding test completed!")

if __name__ == "__main__":
    main()
