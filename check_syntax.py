#!/usr/bin/env python
"""Syntax checker for Python files using ast module."""

import ast
import sys
from pathlib import Path

files = [
    'tests/conftest.py',
    'tests/test_cpu_validation.py',
    'tests/test_gpu_validation.py',
    'tests/test_memory_validation.py',
    'tests/test_interconnect_validation.py',
    'tests/test_bottleneck_analysis.py'
]

print("=" * 70)
print("PYTHON SYNTAX VALIDATION REPORT")
print("=" * 70)

passed = 0
failed = 0

for filepath in files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        print(f"✓ {filepath}: PASS")
        passed += 1
    except SyntaxError as e:
        print(f"✗ {filepath}: FAIL")
        print(f"  Error: Syntax error at line {e.lineno}")
        print(f"  Message: {e.msg}")
        if e.text:
            print(f"  Code: {e.text.strip()}")
            if e.offset:
                print(f"        {' ' * (e.offset - 1)}^")
        failed += 1
    except FileNotFoundError:
        print(f"✗ {filepath}: FILE NOT FOUND")
        failed += 1
    except Exception as e:
        print(f"✗ {filepath}: ERROR")
        print(f"  {type(e).__name__}: {e}")
        failed += 1

print("=" * 70)
print(f"SUMMARY: {passed} passed, {failed} failed out of {len(files)} files")
print("=" * 70)

sys.exit(0 if failed == 0 else 1)
