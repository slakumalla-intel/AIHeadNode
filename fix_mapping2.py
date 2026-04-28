import csv
from pathlib import Path
import shutil

csv_file = Path('artifacts/reports/finalized_test_to_testcaseid_map.csv')
backup_file = Path('artifacts/reports/finalized_test_to_testcaseid_map_backup.csv')
temp_file = Path('artifacts/reports/finalized_test_to_testcaseid_map_temp.csv')

# Read current mapping
rows = []
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Create new rows for parametrized PERFORMANCE tests
perf_rows = []
for i in range(1, 26):
    case_id = f'AI_HOSTCPU_PERF_TEST_ID_{i:03d}'
    selector = f'tests/test_performance_suite.py::test_performance_case_suite[{case_id}]'
    perf_rows.append({
        'TestSelector': selector,
        'File': 'tests/test_performance_suite.py',
        'Class': '',
        'TestName': 'test_performance_case_suite',
        'FinalPrimaryTestCaseID': case_id,
        'FinalCandidateTestCaseIDs': case_id,
        'MappingSource': 'EXACT_CATALOG',
        'Confidence': 'EXACT',
        'FinalizationStatus': 'FINALIZED',
        'OriginalMapped': 'YES',
        'OriginalMappedTestCaseIDs': case_id,
        'Notes': f'Performance suite case {i}'
    })

# Create new rows for parametrized MLPERF tests
ml_rows = []
for i in range(1, 22):
    case_id = f'AI_HOSTCPU_ML_TEST_ID_{i:03d}'
    selector = f'tests/test_mlperf_suite.py::test_mlperf_case_suite[{case_id}]'
    ml_rows.append({
        'TestSelector': selector,
        'File': 'tests/test_mlperf_suite.py',
        'Class': '',
        'TestName': 'test_mlperf_case_suite',
        'FinalPrimaryTestCaseID': case_id,
        'FinalCandidateTestCaseIDs': case_id,
        'MappingSource': 'EXACT_CATALOG',
        'Confidence': 'EXACT',
        'FinalizationStatus': 'FINALIZED',
        'OriginalMapped': 'YES',
        'OriginalMappedTestCaseIDs': case_id,
        'Notes': f'MLPerf suite case {i}'
    })

# Combine all rows
all_rows = rows + perf_rows + ml_rows

# Write to temp file
with open(temp_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(all_rows)

# Backup original and replace with temp
shutil.move(str(csv_file), str(backup_file))
shutil.move(str(temp_file), str(csv_file))

print(f'Updated mapping:')
print(f'  Original rows: {len(rows)}')
print(f'  Added PERFORMANCE parametrized: {len(perf_rows)}')
print(f'  Added MLPERF parametrized: {len(ml_rows)}')
print(f'  Total rows: {len(all_rows)}')
print(f'  Backup saved to: {backup_file}')
