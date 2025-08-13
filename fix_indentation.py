#!/usr/bin/env python3
"""
Indentation fixer untuk jnm_GAN_AHTR.py
Script ini akan memperbaiki masalah indentasi secara otomatis
"""

def fix_indentation():
    """Fix indentation issues in jnm_GAN_AHTR.py"""
    
    input_file = 'jnm_GAN_AHTR.py'
    output_file = 'jnm_GAN_AHTR_fixed.py'
    
    print(f"Fixing indentation in {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_function = False
    function_indent = 0
    
    for i, line in enumerate(lines):
        line_num = i + 1
        original_line = line
        
        # Check if this is a function definition
        stripped = line.strip()
        
        if stripped.startswith('def ') and ':' in stripped:
            in_function = True
            function_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            continue
        
        # Special fixes for known problematic lines
        if line_num == 318:  # grey_image.save line
            fixed_lines.append('\tgrey_image.save("deg_image2.png")\n')
            continue
        elif line_num == 319:  # deg_image = plt.imread line
            fixed_lines.append('\tdeg_image = plt.imread("deg_image2.png")\n')
            continue
        elif line_num == 321:  # gt_image_path line
            fixed_lines.append('\tgt_image_path = os.path.join(DatabasePath, split, "images", im_name)\n')
            continue
        elif line_num == 322:  # original_image = Image.open line
            fixed_lines.append('\toriginal_image = Image.open(gt_image_path)\n')
            continue
        
        # Handle lines that are clearly inside functions but have wrong indentation
        if in_function and stripped:
            if not stripped.startswith('def ') and not stripped.startswith('class '):
                # If line has content but wrong indentation, fix it
                if line.startswith('\t') or line.startswith('    '):
                    # Already indented, keep as is
                    fixed_lines.append(line)
                elif not line.startswith(' ') and stripped:
                    # No indentation but should be indented
                    fixed_lines.append('\t' + stripped + '\n')
                else:
                    fixed_lines.append(line)
            else:
                # New function definition
                in_function = True
                function_indent = len(line) - len(line.lstrip())
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
        
        # Check if we're leaving a function
        if in_function and stripped and not line.startswith(' ') and not line.startswith('\t'):
            if not stripped.startswith('def ') and not stripped.startswith('class '):
                in_function = False
    
    # Write fixed file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed file saved as {output_file}")
    
    # Test compilation
    try:
        import py_compile
        py_compile.compile(output_file, doraise=True)
        print("✅ Fixed file compiles successfully!")
        
        # Replace original file
        import shutil
        shutil.copy(output_file, input_file)
        print(f"✅ Original file {input_file} updated")
        
    except py_compile.PyCompileError as e:
        print(f"❌ Compilation error: {e}")
        print("Manual fixes may be needed")

if __name__ == "__main__":
    fix_indentation()
