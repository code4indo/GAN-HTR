#!/usr/bin/env python3
"""
Validate and fix the network/model.py file
"""

import ast
import os
import sys
from pathlib import Path

def validate_python_syntax(file_path):
    """
    Validate Python syntax and provide specific error information
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse the AST
        ast.parse(content)
        print(f"✅ {file_path} has valid Python syntax")
        return True, None
        
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}:")
        print(f"   Line {e.lineno}: {e.msg}")
        if e.text:
            print(f"   Text: {e.text.strip()}")
        return False, e
        
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False, e

def fix_common_indentation_issues(file_path):
    """
    Fix common indentation issues
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fixed_lines = []
        changes_made = False
        
        for i, line in enumerate(lines):
            original_line = line
            
            # Convert tabs to spaces
            if '\t' in line:
                line = line.expandtabs(4)
                changes_made = True
            
            # Fix lines that start with unexpected indentation
            if line.strip() and line.startswith(' '):
                # Check if this line should be indented based on previous context
                if i > 0:
                    prev_line = lines[i-1].strip() if i > 0 else ""
                    
                    # If previous line doesn't suggest indentation is needed
                    if prev_line and not (prev_line.endswith(':') or 
                                        prev_line.endswith('\\') or
                                        '(' in prev_line and ')' not in prev_line):
                        
                        # Check if this is part of a function/class definition
                        stripped = line.strip()
                        if not any(keyword in stripped for keyword in [
                            'def ', 'class ', 'if ', 'elif ', 'else:', 
                            'for ', 'while ', 'try:', 'except', 'finally:', 
                            'with ', 'return ', 'yield ', 'raise ', 'pass',
                            'break', 'continue'
                        ]):
                            # This might be incorrectly indented
                            print(f"⚠️ Line {i+1} might have incorrect indentation: {stripped[:50]}...")
                            
                            # Try to determine correct indentation
                            # Look for the last line that starts at column 0 or has proper structure
                            proper_indent = 0
                            for j in range(i-1, -1, -1):
                                check_line = lines[j]
                                if check_line.strip():
                                    if check_line.startswith(('def ', 'class ')):
                                        proper_indent = 4
                                        break
                                    elif check_line.rstrip().endswith(':'):
                                        proper_indent = len(check_line) - len(check_line.lstrip()) + 4
                                        break
                                    elif not check_line.startswith(' '):
                                        proper_indent = 0
                                        break
                            
                            # Apply the fix if different from current
                            current_indent = len(line) - len(line.lstrip())
                            if current_indent != proper_indent:
                                line = ' ' * proper_indent + line.lstrip()
                                changes_made = True
                                print(f"   🔧 Fixed indentation: {current_indent} -> {proper_indent}")
            
            fixed_lines.append(line)
        
        if changes_made:
            # Backup original file
            backup_path = f"{file_path}.backup"
            os.rename(file_path, backup_path)
            print(f"📦 Original file backed up to: {backup_path}")
            
            # Write fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            
            print(f"✅ Applied indentation fixes to {file_path}")
            return True
        else:
            print(f"ℹ️ No indentation fixes needed for {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing indentation: {e}")
        return False

def main():
    """Main validation function"""
    model_file = Path("network/model.py")
    
    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        return False
    
    print(f"🔍 Validating {model_file}...")
    
    # First check syntax
    is_valid, error = validate_python_syntax(model_file)
    
    if not is_valid and isinstance(error, SyntaxError):
        if "unexpected indent" in str(error.msg).lower():
            print(f"\n🔧 Attempting to fix indentation issues...")
            
            if fix_common_indentation_issues(model_file):
                print(f"\n🔍 Re-validating after fixes...")
                is_valid, _ = validate_python_syntax(model_file)
                
                if is_valid:
                    print("✅ Indentation issues resolved!")
                    return True
                else:
                    print("❌ Additional issues remain")
            else:
                print("❌ Could not automatically fix indentation")
        
        # Manual fix suggestion for line 480
        if error.lineno == 480:
            print(f"\n💡 Manual fix suggestion for line 480:")
            print(f"   The line 'output_data = Dense(units=d_model, activation=\"linear\")(bgru)'")
            print(f"   likely has incorrect indentation. It should probably be:")
            print(f"   - At the same level as other statements in the function")
            print(f"   - Typically indented with 4 or 8 spaces from the start")
            
    return is_valid

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
