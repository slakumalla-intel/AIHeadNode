# KPI Status Report

- Latest records considered: 182
- Findings detected: 12

## Findings By Reason

| Reason | Count |
|---|---:|
| scaling_ratio_below_required | 2 |
| directional_symmetry_below_required | 2 |
| power_variance_high | 2 |
| failing_gpus_nonempty | 2 |
| below_threshold_gbs | 2 |
| below_min_threshold_gbs | 2 |

## Findings (first 30)

| ID | File | Reason | Details |
|---|---|---|---|
| test_d2h_bandwidth_all_gpus | kpi_from_validation_full_log.jsonl | failing_gpus_nonempty | {"count":1} |
| test_d2h_bandwidth_all_gpus | kpi_from_validation_full_log.jsonl | below_threshold_gbs | {"threshold_gbs":40.95,"metric":"min_gbs","value":39.02} |
| test_p2p_bandwidth_latency_binary | kpi_from_validation_full_log.jsonl | below_min_threshold_gbs | {"metric":"unidir_min_gbs","min_threshold_gbs":40,"value":37.71} |
| test_p2p_multi_stream_bandwidth | kpi_from_validation_full_log.jsonl | scaling_ratio_below_required | {"observed":0.633,"required":0.85} |
| test_p2p_multi_stream_bandwidth | kpi_from_validation_full_log.jsonl | directional_symmetry_below_required | {"observed":0.63,"required":0.75} |
| test_sustained_gpu_power_stability | kpi_from_validation_full_log.jsonl | power_variance_high | {"std_w":57.67,"expected_max":50} |
| test_p2p_bandwidth_latency_binary | node_validation_kpis.jsonl | below_min_threshold_gbs | {"metric":"unidir_min_gbs","min_threshold_gbs":40,"value":37.71} |
| test_d2h_bandwidth_all_gpus | node_validation_kpis.jsonl | failing_gpus_nonempty | {"count":1} |
| test_d2h_bandwidth_all_gpus | node_validation_kpis.jsonl | below_threshold_gbs | {"threshold_gbs":40.95,"metric":"min_gbs","value":39.02} |
| test_p2p_multi_stream_bandwidth | node_validation_kpis.jsonl | scaling_ratio_below_required | {"observed":0.633,"required":0.85} |
| test_p2p_multi_stream_bandwidth | node_validation_kpis.jsonl | directional_symmetry_below_required | {"observed":0.63,"required":0.75} |
| test_sustained_gpu_power_stability | node_validation_kpis.jsonl | power_variance_high | {"std_w":57.67,"expected_max":50} |
