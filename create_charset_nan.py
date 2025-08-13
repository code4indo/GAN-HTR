#!/usr/bin/env python3
"""
Script untuk membuat charset_base (CHAR_LIST) baru dari dataset NaN
Mengekstrak semua kata unik dari ground truth text
"""

import os
import re
from collections import Counter

def normalize_text(text):
    """
    Normalize text untuk konsistensi
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_words_from_dataset():
    """
    Extract semua kata unik dari dataset NaN
    """
    all_words = set()
    
    # Process training set
    splits = ['train', 'test', 'validation']
    
    for split in splits:
        lines_file = f'datasets/nan_raw_biner/{split}/lines.txt'
        
        if os.path.exists(lines_file):
            print(f"Processing {split} set...")
            
            with open(lines_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) >= 2:
                            filename = parts[0]
                            text = parts[1]
                            
                            # Normalize text
                            normalized_text = normalize_text(text)
                            
                            # Split into words
                            words = normalized_text.split()
                            
                            # Add words to set
                            for word in words:
                                # Filter out empty words and very short words
                                if len(word) >= 1:
                                    all_words.add(word)
                        
                        if line_num % 1000 == 0:
                            print(f"  Processed {line_num} lines, {len(all_words)} unique words so far")
    
    return sorted(all_words)

def create_charset_file(words, output_file='Sets/CHAR_LIST_NAN'):
    """
    Create CHAR_LIST file dengan format yang dibutuhkan
    """
    # Ensure Sets directory exists
    os.makedirs('Sets', exist_ok=True)
    
    print(f"Creating charset file with {len(words)} unique words...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Add special tokens
        f.write('<UNK>\n')    # Unknown word token
        f.write('<SPACE>\n')  # Space token
        f.write('|\n')        # Separator token
        
        # Add all unique words
        for word in words:
            f.write(f'{word}\n')
    
    print(f"Charset file created: {output_file}")
    print(f"Total tokens: {len(words) + 3}")  # +3 for special tokens

def analyze_dataset_stats(words):
    """
    Analyze dataset statistics
    """
    print("\n=== Dataset Analysis ===")
    print(f"Total unique words: {len(words)}")
    print(f"Shortest word: '{min(words, key=len)}' ({len(min(words, key=len))} chars)")
    print(f"Longest word: '{max(words, key=len)}' ({len(max(words, key=len))} chars)")
    
    # Word length distribution
    word_lengths = [len(word) for word in words]
    avg_length = sum(word_lengths) / len(word_lengths)
    print(f"Average word length: {avg_length:.2f} characters")
    
    # Most common starting letters
    first_letters = [word[0] for word in words if word]
    letter_counts = Counter(first_letters)
    print(f"Most common starting letters: {letter_counts.most_common(10)}")

def main():
    print("Extracting words from NaN dataset...")
    
    # Extract all unique words
    unique_words = extract_words_from_dataset()
    
    # Analyze statistics
    analyze_dataset_stats(unique_words)
    
    # Create charset file
    create_charset_file(unique_words)
    
    # Also create a backup of current CHAR_LIST if exists
    if os.path.exists('Sets/CHAR_LIST'):
        os.rename('Sets/CHAR_LIST', 'Sets/CHAR_LIST_BACKUP')
        print("Backed up original CHAR_LIST to CHAR_LIST_BACKUP")
    
    # Copy new charset as main CHAR_LIST
    if os.path.exists('Sets/CHAR_LIST_NAN'):
        with open('Sets/CHAR_LIST_NAN', 'r', encoding='utf-8') as src:
            with open('Sets/CHAR_LIST', 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print("New CHAR_LIST created successfully!")
    
    print(f"\n✅ Charset creation completed!")
    print(f"📁 Files created:")
    print(f"   - Sets/CHAR_LIST (main charset)")
    print(f"   - Sets/CHAR_LIST_NAN (NaN-specific charset)")
    print(f"   - Sets/CHAR_LIST_BACKUP (backup of original)")

if __name__ == "__main__":
    main()
