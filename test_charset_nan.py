#!/usr/bin/env python3
"""
Test script untuk menguji charset baru dengan dataset NaN
"""

import os
import sys

# Add current directory to path
sys.path.append('.')

def read_file_char(file_path):
    """Read character list from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def encode_txt(text, charset_base):
    """Encode text using charset"""
    encoded = []
    words = text.split()
    
    for word in words:
        try:
            index = charset_base.index(word.lower())
            encoded.append(index)
        except ValueError:
            # Use <UNK> token for unknown words
            unk_index = charset_base.index('<UNK>')
            encoded.append(unk_index)
            print(f"Warning: Unknown word '{word}' replaced with <UNK>")
    
    return encoded

def test_charset():
    """Test charset dengan sample text dari dataset NaN"""
    
    # Load charset
    charset_base = read_file_char('Sets/CHAR_LIST')
    print(f"Loaded charset with {len(charset_base)} tokens")
    print(f"First 10 tokens: {charset_base[:10]}")
    
    # Test dengan sample text dari dataset
    sample_texts = [
        "Rombouw zijne onderdanen over haer begaen Schermstuck",
        "jegens onsen staet te straffen, maer deselve ter contrarie",
        "uijtgelaten worden, dat onderstaen op Malacca te roven"
    ]
    
    print("\n=== Testing Text Encoding ===")
    for i, text in enumerate(sample_texts, 1):
        print(f"\nTest {i}: {text}")
        
        # Normalize text (lowercase)
        normalized_text = text.lower()
        print(f"Normalized: {normalized_text}")
        
        # Encode
        encoded = encode_txt(normalized_text, charset_base)
        print(f"Encoded length: {len(encoded)}")
        print(f"Encoded (first 10): {encoded[:10]}")
        
        # Decode back to verify
        decoded_words = []
        for idx in encoded:
            if idx < len(charset_base):
                decoded_words.append(charset_base[idx])
        
        decoded_text = ' '.join(decoded_words)
        print(f"Decoded: {decoded_text}")
        
        # Check if encoding is successful
        if decoded_text == normalized_text:
            print("✅ Encoding/Decoding successful!")
        else:
            print("❌ Encoding/Decoding mismatch!")

def test_dataset_coverage():
    """Test coverage of charset terhadap dataset"""
    
    charset_base = read_file_char('Sets/CHAR_LIST')
    charset_words = set(charset_base)
    
    print("\n=== Testing Dataset Coverage ===")
    
    # Test dengan beberapa file ground truth
    lines_file = 'datasets/nan_raw_biner/train/lines.txt'
    
    total_words = 0
    covered_words = 0
    unknown_words = set()
    
    with open(lines_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num >= 100:  # Test first 100 lines only
                break
                
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                if len(parts) >= 2:
                    text = parts[1].lower()
                    words = text.split()
                    
                    for word in words:
                        total_words += 1
                        if word in charset_words:
                            covered_words += 1
                        else:
                            unknown_words.add(word)
    
    coverage = (covered_words / total_words) * 100 if total_words > 0 else 0
    print(f"Words tested: {total_words}")
    print(f"Words covered: {covered_words}")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Unknown words: {len(unknown_words)}")
    
    if unknown_words:
        print(f"Sample unknown words: {list(unknown_words)[:10]}")

def main():
    print("Testing NaN Charset...")
    test_charset()
    test_dataset_coverage()
    print("\n✅ Charset testing completed!")

if __name__ == "__main__":
    main()
