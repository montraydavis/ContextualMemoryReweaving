#!/usr/bin/env python3
"""
Validation script for the reconstruction system.
This script checks the code structure and syntax without requiring PyTorch.
"""

import ast
import os
import sys
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python syntax by parsing the file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the source code
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error parsing file: {e}"

def validate_imports(file_path):
    """Check if the file has valid import statements."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        # Check for import statements
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return True, imports
    except Exception as e:
        return False, f"Error checking imports: {e}"

def validate_class_definitions(file_path):
    """Check if the file has the expected class definitions."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return True, classes
    except Exception as e:
        return False, f"Error checking classes: {e}"

def validate_function_definitions(file_path):
    """Check if the file has the expected function definitions."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        return True, functions
    except Exception as e:
        return False, f"Error checking functions: {e}"

def main():
    """Main validation function."""
    print("🔍 Validating Reconstruction System Code Structure")
    print("=" * 60)
    
    # Files to validate
    files_to_check = [
        "models/reconstruction.py",
        "demo_reconstruction.py",
        "tests/test_reconstruction.py"
    ]
    
    all_passed = True
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            all_passed = False
            continue
        
        print(f"\n📁 Validating: {file_path}")
        print("-" * 40)
        
        # Check syntax
        syntax_valid, syntax_error = validate_python_syntax(file_path)
        if syntax_valid:
            print(f"  ✅ Syntax: Valid")
        else:
            print(f"  ❌ Syntax: {syntax_error}")
            all_passed = False
        
        # Check imports
        imports_valid, imports = validate_imports(file_path)
        if imports_valid:
            print(f"  ✅ Imports: {len(imports)} imports found")
            if imports:
                print(f"     Imported modules: {', '.join(imports[:5])}{'...' if len(imports) > 5 else ''}")
        else:
            print(f"  ❌ Imports: {imports}")
            all_passed = False
        
        # Check classes
        classes_valid, classes = validate_class_definitions(file_path)
        if classes_valid:
            print(f"  ✅ Classes: {len(classes)} classes found")
            if classes:
                print(f"     Classes: {', '.join(classes)}")
        else:
            print(f"  ❌ Classes: {classes}")
            all_passed = False
        
        # Check functions
        functions_valid, functions = validate_function_definitions(file_path)
        if functions_valid:
            print(f"  ✅ Functions: {len(functions)} functions found")
            if functions:
                print(f"     Functions: {', '.join(functions[:5])}{'...' if len(functions) > 5 else ''}")
        else:
            print(f"  ❌ Functions: {functions}")
            all_passed = False
    
    # Check file structure
    print(f"\n📁 File Structure Validation")
    print("-" * 40)
    
    expected_files = [
        "models/reconstruction.py",
        "demo_reconstruction.py", 
        "tests/test_reconstruction.py"
    ]
    
    for expected_file in expected_files:
        if os.path.exists(expected_file):
            file_size = os.path.getsize(expected_file)
            print(f"  ✅ {expected_file}: {file_size} bytes")
        else:
            print(f"  ❌ {expected_file}: Missing")
            all_passed = False
    
    # Check directory structure
    print(f"\n📁 Directory Structure")
    print("-" * 40)
    
    expected_dirs = ["models", "tests", "utils", "experiments"]
    for expected_dir in expected_dirs:
        if os.path.isdir(expected_dir):
            files_in_dir = len([f for f in os.listdir(expected_dir) if f.endswith('.py')])
            print(f"  ✅ {expected_dir}/: {files_in_dir} Python files")
        else:
            print(f"  ❌ {expected_dir}/: Missing")
            all_passed = False
    
    # Summary
    print(f"\n📊 Validation Summary")
    print("=" * 60)
    
    if all_passed:
        print("🎉 All validations passed! The reconstruction system is properly structured.")
        print("\n💡 Next steps:")
        print("  1. Install PyTorch: pip install torch")
        print("  2. Run the demo: python3 demo_reconstruction.py")
        print("  3. Run tests: python3 -m pytest tests/")
    else:
        print("❌ Some validations failed. Please check the errors above.")
        print("\n🔧 Issues to fix:")
        print("  - Check file paths and existence")
        print("  - Verify Python syntax")
        print("  - Ensure all required imports are available")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
