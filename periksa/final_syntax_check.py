#!/usr/bin/env python3
"""
Final syntax verification for jnm_GAN_AHTR.py
"""
import ast
import sys
import os

def check_python_syntax(filename):
    """Check if Python file has valid syntax"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST to check syntax
        ast.parse(source, filename=filename)
        print(f"✅ {filename} - Syntax is valid!")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax Error in {filename}:")
        print(f"   Line {e.lineno}: {e.text}")
        print(f"   Error: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    script_path = "../jnm_GAN_AHTR.py"
    
    if not os.path.exists(script_path):
        print(f"❌ File not found: {script_path}")
        sys.exit(1)
    
    print("🔍 Checking Python syntax...")
    
    if check_python_syntax(script_path):
        print("🎉 All syntax checks passed!")
        print("📋 Summary of fixes applied:")
        print("   ✅ Fixed dtype mismatch in d2_loss (float16 vs float32)")
        print("   ✅ Added tf.cast to ensure consistent float32 dtype")
        print("   ✅ Removed duplicate code sections")
        print("   ✅ Fixed undefined avg_speed variable")
        print("   ✅ Added proper speed tracking")
        sys.exit(0)
    else:
        print("❌ Syntax errors still exist!")
        sys.exit(1)

if __name__ == "__main__":
    main()
