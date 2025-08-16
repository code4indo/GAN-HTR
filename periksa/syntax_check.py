#!/usr/bin/env python3
"""
Quick syntax check for jnm_GAN_AHTR.py
"""
import sys
import ast

def check_syntax(filename):
    try:
        with open(filename, 'r') as f:
            source = f.read()
        
        # Try to parse the AST
        ast.parse(source)
        print(f"✅ Syntax check passed for {filename}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {filename}:")
        print(f"Line {e.lineno}: {e.text}")
        print(f"Error: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error checking {filename}: {e}")
        return False

if __name__ == "__main__":
    check_syntax("../jnm_GAN_AHTR.py")
