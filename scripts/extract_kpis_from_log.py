#!/usr/bin/env python3
"""Extract KPI metrics from alltests.log and generate fresh KPI_VALUES.csv"""

import csv
import re
from pathlib import Path
from collections import defaultdict

LOG_FILE = Path("alltests.log")
KPI_OUTPUT = Path("artifacts/KPI_VALUES.csv")

def extract_kpis_from_log(log_path):
    """Extract KPI measurements from test log"""
    content = log_path.read_text()
    kpis = []
    
    # Pattern: KPI [test_name]: {metrics}
    kpi_pattern = r'KPI \[([^\]]+)\]:\s*(\{[^}]+\})'
    matches = re.findall(kpi_pattern, content)
    
    for test_name, json_str in matches:
        # Parse the JSON-like dict string
        metrics_str = json_str.strip('{}')
        for item in metrics_str.split(','):
            if ':' in item:
                key, val = item.split(':', 1)
                key = key.strip().strip('"\'')
                val = val.strip().strip('"\'')
                try:
                    val_float = float(val)
                    kpis.append({'metric': key, 'value': val_float, 'test': test_name})
                except ValueError:
                    pass
    
    # Pattern: metric=value in log output
    log_pattern = r'(\w+)=\s*([\d.]+)'
    for match in re.finditer(log_pattern, content):
        metric, value = match.groups()
        if metric not in ['CUDA_LAUNCH_BLOCKING', 'HF_ENABLE_LARGE_MODELS', 'NCCL_P2P_DISABLE']:
            try:
                val_float = float(value)
                # Extract test name from context (look backward in log)
                pos = match.start()
                context = content[max(0, pos-500):pos]
                test_match = re.search(r'(test_\w+)', context[::-1])
                if test_match:
                    test_name = test_match.group(1)[::-1]
                    kpis.append({'metric': metric, 'value': val_float, 'test': test_name})
            except ValueError:
                pass
    
    return kpis

def main():
    print("Extracting KPIs from alltests.log...")
    kpis = extract_kpis_from_log(LOG_FILE)
    
    # Deduplicate and aggregate
    seen = set()
    unique_kpis = []
    for kpi in kpis:
        key = (kpi['test'], kpi['metric'])
        if key not in seen:
            seen.add(key)
            unique_kpis.append(kpi)
    
    print(f"Extracted {len(unique_kpis)} unique KPI records from alltests.log")
    
    # Write to CSV
    KPI_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with KPI_OUTPUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'value', 'test'])
        writer.writeheader()
        for kpi in unique_kpis:
            writer.writerow({
                'metric': kpi['metric'],
                'value': kpi['value'],
                'test': kpi['test']
            })
    
    print(f"Wrote fresh KPI_VALUES.csv with {len(unique_kpis)} rows")

if __name__ == '__main__':
    main()
