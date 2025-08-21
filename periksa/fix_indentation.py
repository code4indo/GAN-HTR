#!/usr/bin/env python3
"""
Script to detect and fix indentation issues in Python files
"""

import os
import sys
import re
from pathlib import Path

def detect_indentation_issues(file_path):
    """
    Detect indentation issues in a Python file
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Check for mixed tabs and spaces
            if '\t' in line and ' ' in line[:len(line) - len(line.lstrip())]:
                issues.append(f"Line {i}: Mixed tabs and spaces")
            
            # Check for unexpected indentation
            if line.startswith(' ') or line.startswith('\t'):
                # This is an indented line - check if it should be
                prev_line_idx = i - 2
                while prev_line_idx >= 0:
                    prev_line = lines[prev_line_idx].strip()
                    if prev_line:
                        break
                    prev_line_idx -= 1
                
                if prev_line_idx >= 0:
                    prev_line = lines[prev_line_idx]
                    # Check if previous line ends with colon or backslash
                    if not (prev_line.rstrip().endswith(':') or 
                           prev_line.rstrip().endswith('\\') or
                           prev_line.strip().endswith('(') or
                           '(' in prev_line and ')' not in prev_line):
                        # Check if this line is a continuation of a previous construct
                        if not any(keyword in stripped for keyword in ['def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ']):
                            issues.append(f"Line {i}: Possibly unexpected indentation: '{stripped[:50]}...'")
    
    except Exception as e:
        issues.append(f"Error reading file: {e}")
    
    return issues

def fix_indentation_issue_at_line(file_path, line_number):
    """
    Fix indentation issue at a specific line
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if line_number <= len(lines):
            target_line = lines[line_number - 1]
            
            # Remove leading whitespace and add proper indentation
            stripped_line = target_line.lstrip()
            
            # Determine proper indentation based on context
            proper_indent = ""
            
            # Look at previous non-empty line for context
            prev_line_idx = line_number - 2
            while prev_line_idx >= 0:
                prev_line = lines[prev_line_idx]
                if prev_line.strip():
                    # Count indentation of previous line
                    prev_indent = len(prev_line) - len(prev_line.lstrip())
                    
                    # If previous line ends with colon, increase indent
                    if prev_line.rstrip().endswith(':'):
                        proper_indent = " " * (prev_indent + 4)
                    else:
                        proper_indent = " " * prev_indent
                    break
                prev_line_idx -= 1
            
            # Apply the fix
            lines[line_number - 1] = proper_indent + stripped_line
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            print(f"✅ Fixed indentation at line {line_number}")
            return True
            
    except Exception as e:
        print(f"❌ Error fixing line {line_number}: {e}")
        return False

def main():
    """Main function to check and fix indentation"""
    
    # Check network/model.py specifically
    model_file = Path("network/model.py")
    
    if not model_file.exists():
        print(f"❌ File not found: {model_file}")
        return
    
    print(f"🔍 Checking indentation in {model_file}...")
    
    issues = detect_indentation_issues(model_file)
    
    if issues:
        print(f"📋 Found {len(issues)} potential issues:")
        for issue in issues:
            print(f"   {issue}")
        
        # Try to fix the specific issue at line 480
        print(f"\n🔧 Attempting to fix line 480...")
        if fix_indentation_issue_at_line(model_file, 480):
            print("✅ Fix applied successfully")
        else:
            print("❌ Could not apply automatic fix")
            
        # Recheck after fix
        print(f"\n🔍 Rechecking after fix...")
        new_issues = detect_indentation_issues(model_file)
        if new_issues:
            print(f"⚠️ {len(new_issues)} issues remain:")
            for issue in new_issues:
                print(f"   {issue}")
        else:
            print("✅ All indentation issues resolved!")
    else:
        print("✅ No indentation issues found")

if __name__ == "__main__":
    main()
