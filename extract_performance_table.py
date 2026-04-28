import csv
from collections import defaultdict

# Read KPI CSV
metrics_by_category = {
    'Functional': [],
    'Inference': [],
    'NVBandwidth': [],
    'P2P': [],
    'Training': []
}

# Map keywords to categories
keyword_map = {
    'Functional': ['correctness', 'correct', 'validation', 'pass', 'fail', 'error', 'event_wait', 'mmio', 'memory_allocation'],
    'Inference': ['latency', 'throughput', 'prefill', 'tokens', 'batch', 'dispatch', 'tflops', 'gflops', 'fps'],
    'NVBandwidth': ['nvlink', 'hbm', 'bandwidth', 'bw_'],
    'P2P': ['p2p', 'peer', 'allgather', 'allreduce', 'nccl'],
    'Training': ['backward', 'forward', 'gradient', 'optimizer', 'loss', 'epoch', 'compile']
}

with open('artifacts/KPI_VALUES.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    seen = set()
    
    for row in reader:
        metric = row['metric'].lower()
        value = row['value']
        test = row['test']
        
        key = (row['metric'], value)
        if key in seen:
            continue
        seen.add(key)
        
        # Categorize
        for category, keywords in keyword_map.items():
            if any(kw in metric for kw in keywords):
                try:
                    val = float(value)
                    metrics_by_category[category].append({
                        'Metric': row['metric'],
                        'Value': val,
                        'Test': test
                    })
                except:
                    pass
                break

# Display results
print("="*120)
print("PERFORMANCE METRICS BY CATEGORY")
print("="*120)

for category in ['Functional', 'Inference', 'NVBandwidth', 'P2P', 'Training']:
    items = metrics_by_category[category]
    if items:
        print(f"\n\n### {category.upper()} PERFORMANCE")
        print("-"*120)
        print(f"{'Metric':<45} {'Value':<18} {'Test':<45}")
        print("-"*120)
        
        # Sort by value descending
        for item in sorted(items, key=lambda x: x['Value'], reverse=True)[:15]:
            print(f"{item['Metric']:<45} {item['Value']:<18.4f} {item['Test']:<45}")
        print(f"\nTotal {category} metrics: {len(items)}")

print("\n" + "="*120)
