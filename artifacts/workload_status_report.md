# Workload Execution Status Report

Generated (UTC): 2026-07-21T09:31:13.047698+00:00
Host: sc00901168s0075

## Summary

- Domains run: 13
- Passed: 4
- Failed: 9
- Total tests: 320
- Failures: 46
- Errors: 2
- Skipped: 29

## Domain Status

| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log |
|---|---|---:|---:|---:|---:|---:|---|---|
| bottleneck | FAIL | 25 | 2 | 0 | 1 | 19.051 | artifacts/reports/junit/bottleneck.xml | artifacts/reports/logs/bottleneck.log |
| cpu | PASS | 33 | 0 | 0 | 0 | 236.795 | artifacts/reports/junit/cpu.xml | artifacts/reports/logs/cpu.log |
| memory | FAIL | 28 | 7 | 0 | 0 | 24.852 | artifacts/reports/junit/memory.xml | artifacts/reports/logs/memory.log |
| gpu | PASS | 37 | 0 | 0 | 0 | 12.443 | artifacts/reports/junit/gpu.xml | artifacts/reports/logs/gpu.log |
| interconnect | FAIL | 41 | 5 | 0 | 5 | 57.106 | artifacts/reports/junit/interconnect.xml | artifacts/reports/logs/interconnect.log |
| node_validation | FAIL | 51 | 7 | 0 | 5 | 2106.07 | artifacts/reports/junit/node_validation.xml | artifacts/reports/logs/node_validation.log |
| agentic_ai | PASS | 9 | 0 | 0 | 0 | 17.58 | artifacts/reports/junit/agentic_ai.xml | artifacts/reports/logs/agentic_ai.log |
| ai_pipeline | FAIL | 1 | 0 | 1 | 0 | 0.181 | artifacts/reports/junit/ai_pipeline.xml | artifacts/reports/logs/ai_pipeline.log |
| system_health | FAIL | 39 | 5 | 0 | 14 | 42.209 | artifacts/reports/junit/system_health.xml | artifacts/reports/logs/system_health.log |
| system_config | PASS | 8 | 0 | 0 | 0 | 3.051 | artifacts/reports/junit/system_config.xml | artifacts/reports/logs/system_config.log |
| vllm_prefill | FAIL | 1 | 0 | 1 | 0 | 0.124 | artifacts/reports/junit/vllm_prefill.xml | artifacts/reports/logs/vllm_prefill.log |
| performance_suite | FAIL | 26 | 1 | 0 | 2 | 4285.791 | artifacts/reports/junit/performance_suite.xml | artifacts/reports/logs/performance_suite.log |
| mlperf_suite | FAIL | 21 | 19 | 0 | 2 | 1878.903 | artifacts/reports/junit/mlperf_suite.xml | artifacts/reports/logs/mlperf_suite.log |

## Failure Details

### bottleneck

- [failure] tests.test_bottleneck_analysis.TestCPUMemoryBandwidthBottleneck::test_cpu_dram_bandwidth_vs_hbm: AssertionError: HBM/DRAM ratio 6.5× < 10× expected – CPU memory BW may bottleneck data preprocessing
assert 6.510416666666667 >= 10.0
- [failure] tests.test_bottleneck_analysis.TestNVLinkBottleneck::test_nvlink_bw_vs_theoretical: AssertionError: NVLink P2P BW GPU 0->1: efficiency 29.32% is below threshold 70.00% (measured=263.918, theoretical=900.000)
assert 0.2932420470545126 >= 0.7

### memory

- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_scale_bw_exceeds_15gbs: AssertionError: STREAM Scale 13.27 GB/s < 15 GB/s
assert 13.272585291518102 >= 15.0
- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_triad_bw_exceeds_15gbs: AssertionError: STREAM Triad 14.00 GB/s < 15 GB/s
assert 13.999563240599834 >= 15.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[32-4.0]: AssertionError: Working-set 32 KB: latency 89.61 ns > 4.0 ns
assert 89.60620112574702 <= 4.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[512-8.0]: AssertionError: Working-set 512 KB: latency 83.19 ns > 8.0 ns
assert 83.18835448273276 <= 8.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[16384-30.0]: AssertionError: Working-set 16384 KB: latency 177.50 ns > 30.0 ns
assert 177.50110000633867 <= 30.0
- [failure] tests.test_memory_validation.TestMemoryLatency::test_latency_at_working_set[524288-120.0]: AssertionError: Working-set 524288 KB: latency 224.87 ns > 120.0 ns
assert 224.87485999590717 <= 120.0
- [failure] tests.test_memory_validation.TestCacheHierarchyBandwidth::test_bandwidth_at_working_set[L3-cache-8.0-20.0]: AssertionError: L3-cache: measured 17.4 GB/s < expected 20.0 GB/s
assert 17.37882306887742 >= 20.0

### interconnect

- [failure] tests.test_interconnect_validation.TestPCIeTopology::test_pcie_gen5_sysfs: AssertionError: No NVIDIA GPU PCIe Gen 5 (32 GT/s) links found. Observed: ['32.0 GT/s PCIe']
assert False
- [failure] tests.test_interconnect_validation.TestNVLinkDetection::test_nvlink_link_count_per_gpu: AttributeError: 'GPUSpecs' object has no attribute 'nvlink_links_per_gpu'. Did you mean: 'fabric_links_per_gpu'?
- [failure] tests.test_interconnect_validation.TestP2PBandwidth::test_p2p_bandwidth_pair[0-1]: AssertionError: NVLink P2P BW GPU 0->1: efficiency 29.43% is below threshold 70.00% (measured=264.845, theoretical=900.000)
assert 0.29427238779567194 >= 0.7
- [failure] tests.test_interconnect_validation.TestNVLinkAllToAll::test_all_to_all_aggregate_bandwidth: AssertionError: All-to-all P2P aggregate BW: efficiency 29.04% is below threshold 70.00% (measured=522.680, theoretical=1800.000)
assert 0.29037764588053 >= 0.7
- [failure] tests.test_interconnect_validation.TestPCIeVsNVLink::test_nvlink_faster_than_pcie: AssertionError: NVLink BW 262.84 GB/s < 5× PCIe BW 54.83 GB/s
assert 262.8442334942865 >= (5 * 54.8252468692978)

### node_validation

- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[deepseek_r1_quantized-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-32-3000.0-True]: Failed: Quantized enterprise workload deepseek-ai/DeepSeek-R1-Distill-Qwen-7B requires 4 visible GPU(s), but only 2 are available. CUDA_VISIBLE_DEVICES=0,1. Memory snapshot: gpu0: free=78.3GiB used=0.9GiB total=79.2GiB; gpu1: free=78.5GiB used=0.6GiB total=79.
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[llama_70b-meta-llama/Meta-Llama-3.1-70B-Instruct-16-12000.0-True]: Failed: Quantized enterprise workload meta-llama/Meta-Llama-3.1-70B-Instruct requires 4 visible GPU(s), but only 2 are available. CUDA_VISIBLE_DEVICES=0,1. Memory snapshot: gpu0: free=78.3GiB used=0.9GiB total=79.2GiB; gpu1: free=78.5GiB used=0.6GiB total=79.2
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[qwen_70b-Qwen/Qwen2.5-72B-Instruct-16-12000.0-True]: Failed: Quantized enterprise workload Qwen/Qwen2.5-72B-Instruct requires 4 visible GPU(s), but only 2 are available. CUDA_VISIBLE_DEVICES=0,1. Memory snapshot: gpu0: free=78.3GiB used=0.9GiB total=79.2GiB; gpu1: free=78.5GiB used=0.6GiB total=79.2GiB
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vllm_data_pipeline_saturation_inference_prefill: AssertionError: GPU utilization not sustained (>=90% ratio=0.00, mean=0.0%, target_ratio=0.50, shared_gpu=True)
assert 0.0 >= 0.5
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vision_object_detection_yolo_fps: Failed: YOLO workload collected 0 worker results for 2 processes
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_llm_training_small_multigpu: Failed: TC-013 worker failures: [{'rank': 0, 'status': 'error', 'error': "Model load failed: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.\nCheck your internet connection or see how to run the li
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_llm_training_large_multigpu: Failed: TC-014 worker failures: [{'rank': 0, 'status': 'error', 'error': 'CUDA out of memory. Tried to allocate 160.00 MiB. GPU 0 has a total capacity of 79.18 GiB of which 102.56 MiB is free. Process 116528 has 884.00 MiB memory in use. Including non-PyTorch 

### ai_pipeline

- [error] tests.test_ai_pipeline_kpis: collection failure

### system_health

- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[0]: AssertionError: GPU 0 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[1]: AssertionError: GPU 1 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle: AssertionError: SM clock dropped 46.2% during sustained load (initial 1755 MHz → min 945 MHz, max_temp 59 °C). Thermal or power-limit throttling detected.
assert 0.46153846153846156 <= 0.1
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_quantised_softmax_output_matches_fp16: AssertionError: Quantised softmax max error 2.56e-03 exceeds 1e-03
assert 0.0025634765625 < 0.001
- [failure] tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager: AssertionError: torch.compile speedup 0.824× < floor 1.05×. Compiled kernels may be suboptimal for this H100 PCIe configuration.
assert 0.8238680050201499 >= 1.05

### vllm_prefill

- [error] tests.test_vllm_prefill_validation: collection failure

### performance_suite

- [failure] tests.test_performance_suite::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_010]: Failed: Command failed for AI_HOSTCPU_PERF_TEST_ID_010: trtllm-bench --model /path/to/hf/Llama-3.3-70B-Instruct throughput --dataset /mnt/datasets/synthetic/synthetic_2048_2048_1000.txt --engine_dir /path/to/engines/baseline
executed_command:
trtllm-bench --mo

### mlperf_suite

- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_003]: Failed: AI_HOSTCPU_ML_TEST_ID_003: command failed (rc=1)
executed_command: vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 --port 8000
stderr_tail:
 a background process. See stack tr
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_004]: Failed: Timeout (>1800.0s) from pytest-timeout.
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_005]: Failed: AI_HOSTCPU_ML_TEST_ID_005: command failed (rc=1)
executed_command: vllm serve meta-llama/Llama-2-70b-chat-hf --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
stderr_tail:
(APIServer pid=150641) Tr
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_006]: Failed: AI_HOSTCPU_ML_TEST_ID_006: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/vision/classification_and_detection/run_local.sh pytorch resnet50 gpu --scenario Offline --dataset-path /mnt/datasets/imagenet --output artifa
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_007]: Failed: AI_HOSTCPU_ML_TEST_ID_007: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/vision/classification_and_detection/run_local.sh pytorch retinanet gpu --scenario Offline --dataset-path /mnt/datasets/openimages --output art
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_008]: Failed: AI_HOSTCPU_ML_TEST_ID_008: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/medical_imaging/3d-unet-kits19/run_local.sh pytorch gpu --scenario Offline --dataset-path /mnt/datasets/kits19 --output artifacts/mlperf_resul
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_009]: Failed: AI_HOSTCPU_ML_TEST_ID_009: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/speech_recognition/rnnt/run_local.sh pytorch rnnt gpu --scenario Offline --dataset-path /mnt/datasets/librispeech --output artifacts/mlperf_re
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_010]: Failed: AI_HOSTCPU_ML_TEST_ID_010: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/speech_recognition/whisper/run_local.sh pytorch gpu --scenario Offline --dataset-path /mnt/datasets/librispeech --output artifacts/mlperf_resu
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_011]: Failed: AI_HOSTCPU_ML_TEST_ID_011: command failed (rc=2)
executed_command: /root/workloads/AIHeadNode/.venv4/bin/python /root/workloads/mlcommons/inference/recommendation/dlrm_v2/pytorch/run.py --scenario Offline --dataset-path /mnt/datasets/criteo --model-pat
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_012]: Failed: AI_HOSTCPU_ML_TEST_ID_012: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/text_to_image/run_local.sh pytorch gpu --scenario Offline --dataset-path /mnt/datasets/coco-sdxl --model-path /mnt/datasets/sdxl --output arti
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_013]: Failed: AI_HOSTCPU_ML_TEST_ID_013: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/graph/R-GAT/run_local.sh pytorch gpu --scenario Offline --dataset-path /mnt/datasets/igb-het --model-path /mnt/datasets/rgat --output artifact
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_014]: Failed: AI_HOSTCPU_ML_TEST_ID_014: command failed (rc=1)
executed_command: torchrun --nproc_per_node=2 /root/workloads/mlcommons/training/language_model/gpt-j/run_training.py --model-name EleutherAI/gpt-j-6b --dataset-path /mnt/datasets/cnn_dailymail --output-
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_015]: Failed: AI_HOSTCPU_ML_TEST_ID_015: command failed (rc=1)
executed_command: torchrun --nproc_per_node=2 /root/workloads/mlcommons/training/language_model/pytorch/bert/run_squad.py --bert_model bert-large-uncased --do_train --train_file /mnt/datasets/squad/train
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_016]: Failed: AI_HOSTCPU_ML_TEST_ID_016: command failed (rc=1)
executed_command: torchrun --nproc_per_node=2 /root/workloads/mlcommons/training/image_segmentation/pytorch/train.py --data-dir /mnt/datasets/kits19 --epochs 4000 --batch-size 2 --optimizer adam --learni
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_017]: Failed: AI_HOSTCPU_ML_TEST_ID_017: command failed (rc=1)
executed_command: torchrun --nproc_per_node=2 /root/workloads/mlcommons/training/stable_diffusion/main.py --base /root/workloads/mlcommons/training/stable_diffusion/configs/train_01x04x08.yaml --actual_r
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_018]: Failed: AI_HOSTCPU_ML_TEST_ID_018: command failed (rc=1)
executed_command: torchrun --nproc_per_node=2 /root/workloads/mlcommons/training/language_model/pytorch/bert/run_pretraining.py --bert_model bert-large-uncased --do_train --train_file /mnt/datasets/bert/
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_019]: Failed: AI_HOSTCPU_ML_TEST_ID_019: command failed (rc=1)
executed_command: torchrun --nproc_per_node=2 /root/workloads/mlcommons/training/large_language_model/megatron-lm/pretrain_gpt.py --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --num-lay
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_020]: Failed: AI_HOSTCPU_ML_TEST_ID_020: command failed (rc=1)
executed_command: torchrun --nproc_per_node=2 /root/workloads/mlcommons/training/llm_finetune/run_lora_finetune.py --model_name_or_path meta-llama/Llama-2-70b-hf --dataset_name scrolls --dataset_config g
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_021]: Failed: AI_HOSTCPU_ML_TEST_ID_021: command failed (rc=1)
executed_command: torchrun --nproc_per_node=2 /root/workloads/mlcommons/training/recommendation/dlrm_v2/pytorch/dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot=13-512-256-128 --arch-mlp-t
