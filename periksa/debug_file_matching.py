#!/usr/bin/env python3
"""
Debug script untuk mengecek nama file enhanced vs list files
"""

import os

def debug_file_matching():
    print("🔍 DEBUG: Checking file matching between enhanced images and lists")
    
    # Check enhanced images
    enhanced_dir = "ResultGanS_iam_OP_debug/epoch3/"
    if os.path.exists(enhanced_dir):
        enhanced_files = [f for f in os.listdir(enhanced_dir) if f.endswith('.png')]
        print(f"\n📁 Enhanced images: {len(enhanced_files)} files")
        
        # Extract base names
        enhanced_base_names = []
        for enhanced_file in enhanced_files[:10]:  # First 10 for debugging
            print(f"   Enhanced file: {enhanced_file}")
            if enhanced_file.endswith('.jpg.png'):
                base_name = enhanced_file[:-8]  # Remove .jpg.png
            elif enhanced_file.endswith('.png'):
                base_name = enhanced_file[:-4]  # Remove .png
            else:
                continue
            enhanced_base_names.append(base_name)
            print(f"   → Base name: {base_name}")
            
            # Remove numeric prefix
            if '_' in base_name and base_name.split('_')[0].isdigit():
                clean_base_name = '_'.join(base_name.split('_')[1:])
                enhanced_base_names.append(clean_base_name)
                print(f"   → Clean base name: {clean_base_name}")
        
        print(f"\n📄 Enhanced base names (first 10):")
        for i, name in enumerate(enhanced_base_names):
            print(f"   {i+1:2d}. {name}")
    
    # Check list files
    list_files = ['Sets/list_test_nan.txt', 'Sets/list_train_nan.txt', 'Sets/list_valid_nan.txt']
    
    for list_file in list_files:
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                list_names = [line.strip() for line in f.readlines() if line.strip()]
            
            print(f"\n📄 {list_file}: {len(list_names)} files")
            print(f"   First 5 entries:")
            for i, name in enumerate(list_names[:5]):
                print(f"   {i+1}. {name}")
            
            # Check intersection
            if 'enhanced_base_names' in locals():
                intersection = []
                for enhanced_name in enhanced_base_names:
                    if enhanced_name in list_names:
                        intersection.append(enhanced_name)
                
                print(f"   ✅ Intersection: {len(intersection)} files")
                if intersection:
                    print(f"   Examples: {intersection[:3]}")
                else:
                    print("   ❌ No intersection found")
                    
                    # Show why no match
                    print("   🔍 Checking first enhanced name in this list:")
                    first_enhanced = enhanced_base_names[0] if enhanced_base_names else None
                    if first_enhanced:
                        found = first_enhanced in list_names
                        print(f"   '{first_enhanced}' in list: {found}")
                        
                        # Find similar names
                        similar = [name for name in list_names[:100] if any(part in name for part in first_enhanced.split('_')[:3])]
                        if similar:
                            print(f"   Similar names in list: {similar[:3]}")

if __name__ == '__main__':
    debug_file_matching()
