import json
import glob

kpi_files = glob.glob('artifacts/*_kpis.jsonl')
all_data = []

for f in sorted(kpi_files):
    try:
        with open(f, encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    record = json.loads(line)
                    if 'metrics' in record:
                        for k, v in record['metrics'].items():
                            # Only include numeric values and non-empty metrics
                            if isinstance(v, (int, float)) and v is not None:
                                all_data.append({
                                    'test': record.get('test', 'unknown'),
                                    'metric': k,
                                    'value': v
                                })
    except Exception as e:
        pass

# Sort by metric name and display
print("=" * 90)
print(f"{'METRIC':<40} {'VALUE':<20} TEST")
print("=" * 90)
for item in sorted(all_data, key=lambda x: x['metric']):
    print(f"{item['metric']:<40} {item['value']:<20} {item['test']}")
print("=" * 90)
print(f"\nTotal KPI numeric values captured: {len(all_data)}")
