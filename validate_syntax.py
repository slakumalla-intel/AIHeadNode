import ast, sys
files = [
    'tests/conftest.py',
    'tests/test_cpu_validation.py',
    'tests/test_gpu_validation.py',
    'tests/test_memory_validation.py',
    'tests/test_interconnect_validation.py',
    'tests/test_bottleneck_analysis.py',
]
all_ok = True
for f in files:
    try:
        with open(f) as fh:
            src = fh.read()
        ast.parse(src)
        print('OK  ' + f)
    except SyntaxError as e:
        print('ERR ' + f + ': ' + str(e))
        all_ok = False
sys.exit(0 if all_ok else 1)
