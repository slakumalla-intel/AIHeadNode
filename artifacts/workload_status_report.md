# Workload Execution Status Report

Generated (UTC): 2026-05-21T11:24:22.180183+00:00
Host: sc00901168s0075

## Summary

- Domains run: 13
- Passed: 4
- Failed: 9
- Total tests: 411
- Failures: 47
- Errors: 0
- Skipped: 38

## Domain Status

| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log |
|---|---|---:|---:|---:|---:|---:|---|---|
| bottleneck | FAIL | 25 | 2 | 0 | 0 | 37.417 | artifacts/reports/junit/bottleneck.xml | artifacts/reports/logs/bottleneck.log |
| cpu | PASS | 33 | 0 | 0 | 1 | 277.108 | artifacts/reports/junit/cpu.xml | artifacts/reports/logs/cpu.log |
| memory | FAIL | 28 | 2 | 0 | 0 | 24.542 | artifacts/reports/junit/memory.xml | artifacts/reports/logs/memory.log |
| gpu | FAIL | 53 | 1 | 0 | 0 | 17.642 | artifacts/reports/junit/gpu.xml | artifacts/reports/logs/gpu.log |
| interconnect | PASS | 77 | 0 | 0 | 4 | 93.055 | artifacts/reports/junit/interconnect.xml | artifacts/reports/logs/interconnect.log |
| node_validation | FAIL | 51 | 8 | 0 | 8 | 1616.093 | artifacts/reports/junit/node_validation.xml | artifacts/reports/logs/node_validation.log |
| agentic_ai | FAIL | 9 | 1 | 0 | 0 | 17.806 | artifacts/reports/junit/agentic_ai.xml | artifacts/reports/logs/agentic_ai.log |
| ai_pipeline | FAIL | 38 | 6 | 0 | 12 | 31.124 | artifacts/reports/junit/ai_pipeline.xml | artifacts/reports/logs/ai_pipeline.log |
| system_health | FAIL | 39 | 7 | 0 | 3 | 44.633 | artifacts/reports/junit/system_health.xml | artifacts/reports/logs/system_health.log |
| system_config | PASS | 8 | 0 | 0 | 0 | 3.511 | artifacts/reports/junit/system_config.xml | artifacts/reports/logs/system_config.log |
| vllm_prefill | PASS | 3 | 0 | 0 | 1 | 629.616 | artifacts/reports/junit/vllm_prefill.xml | artifacts/reports/logs/vllm_prefill.log |
| performance_suite | FAIL | 26 | 1 | 0 | 7 | 1542.034 | artifacts/reports/junit/performance_suite.xml | artifacts/reports/logs/performance_suite.log |
| mlperf_suite | FAIL | 21 | 19 | 0 | 2 | 1842.738 | artifacts/reports/junit/mlperf_suite.xml | artifacts/reports/logs/mlperf_suite.log |

## Failure Details

### bottleneck

- [failure] tests.test_bottleneck_analysis.TestCPUComputeBottleneck::test_cpu_fp32_efficiency_report: AssertionError: CPU FP32 (1 processes × 32 threads): efficiency 3.67% is below threshold 5.00% (measured=0.999, theoretical=27.238)
assert 0.03669401508547574 >= 0.05
- [failure] tests.test_bottleneck_analysis.TestGPUComputeBottleneck::test_per_gpu_fp16_efficiency: AssertionError: GPU 2 FP16 Tensor Core: efficiency 58.04% is below threshold 60.00% (measured=438.810, theoretical=756.000)
assert 0.5804362653412989 >= 0.6

### memory

- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_scale_bw_exceeds_15gbs: AssertionError: STREAM Scale 12.89 GB/s < 15 GB/s
assert 12.88683558344393 >= 15.0
- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_triad_bw_exceeds_15gbs: AssertionError: STREAM Triad 12.88 GB/s < 15 GB/s
assert 12.875201402266258 >= 15.0

### gpu

- [failure] tests.test_gpu_validation.TestHBMBandwidth::test_hbm_bandwidth_per_gpu[0]: AssertionError: HBM3 bandwidth GPU 0: efficiency 17.69% is below threshold 80.00% (measured=0.354, theoretical=2.000)
assert 0.17689374103622998 >= 0.8

### node_validation

- [failure] tests.test_node_validation.TestGPUP2PBandwidth::test_p2p_bandwidth_pytorch_fallback: Failed: P2P bandwidth fallback validation failed.
threshold=40.0 GB/s, measured_pairs=12, inaccessible_pairs=0
Failures:
GPU0->GPU1: 38.1 GB/s < 40.0 GB/s
GPU2->GPU1: 34.5 GB/s < 40.0 GB/s
- [failure] tests.test_node_validation.TestCPUGPUPowerAndThrottling::test_sustained_gpu_power_stability: AssertionError: Power variance 59.3 W is too high (expected < 50 W)
assert np.float64(59.30360696145631) < 50
- [failure] tests.test_node_validation.TestNCCLAndNVLinkValidation::test_nccl_p2p_perf: AssertionError: NCCL sendrecv busbw 24.1 GB/s too low
assert 24.08 >= 30.0
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[llama_70b-meta-llama/Meta-Llama-3.1-70B-Instruct-16-12000.0-True]: ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vllm_data_pipeline_saturation_inference_prefill: AssertionError: GPU utilization not sustained (>=90% ratio=0.00, mean=0.0%, target_ratio=0.50, shared_gpu=True)
assert 0.0 >= 0.5
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vision_object_detection_yolo_fps: AssertionError: YOLO aggregate FPS too low: 97.0 < target 200.0
assert 96.99256190071503 >= 200.0
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_predictable_memory_orchestration_kv_cache_offload: AssertionError: Offload throughput degradation too high: 87.5% (baseline 339.7 -> offload 42.4)
assert 87.51046593768514 <= 60
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_llm_training_small_multigpu: Failed: TC-013 worker failures: [{'rank': 3, 'status': 'error', 'error': "Model load failed: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.\nCheck your internet connection or see how to run the li

### agentic_ai

- [failure] tests.test_agentic_ai_validation.TestToolInvocationOverhead::test_parallel_tool_dispatch_scalability: AssertionError: Tool dispatch scales poorly: 16 tools take 7.0× time of 1 tool
assert 7.029355897081678 < 5.0

### ai_pipeline

- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_tokenizer_multi_thread_scaling: AssertionError: 4-thread tokenizer scaling 1.47× < 2.0× single-thread
assert 1.4748150802127233 >= 2.0
- [failure] tests.test_ai_pipeline_kpis.TestDataLoadingAndPreprocessing::test_async_prefetch_gpu_overlap_efficiency: AssertionError: Async prefetch overlap efficiency 25.0% < 60.0%
assert 24.999455191166728 >= 60.0
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_model_load_time_to_gpu: AssertionError: Model load bandwidth 0.42 GB/s < 1.0 GB/s
assert 0.41791630508399996 >= 1.0
- [failure] tests.test_ai_pipeline_kpis.TestModelOrchestration::test_tensor_parallel_allreduce_overhead_fraction: AssertionError: AllReduce overhead 98.6% > 20.0% of compute
assert 98.6229079369924 <= 20.0
- [failure] tests.test_ai_pipeline_kpis.TestNetworkingAndStorageIO::test_model_weight_load_pipeline_bandwidth: AssertionError: Weight load pipeline 0.87 GB/s < 1.0 GB/s
assert 0.8731330767871053 >= 1.0
- [failure] tests.test_ai_pipeline_kpis.TestInferencePipelineKPIs::test_time_to_first_token[2048]: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API

### system_health

- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[0]: AssertionError: GPU 0 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[1]: AssertionError: GPU 1 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[2]: AssertionError: GPU 2 clock ratio 19.33% below 80%; throttling suspected (current 345 MHz, max 1785 MHz)
assert 0.19327731092436976 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_gpu_clock_boost_ratio[3]: AssertionError: GPU 3 clock ratio 19.66% below 80%; throttling suspected (current 345 MHz, max 1755 MHz)
assert 0.19658119658119658 >= 0.8
- [failure] tests.test_system_health_and_reliability.TestGPUThermalAndClockStability::test_sustained_compute_no_thermal_throttle: AssertionError: SM clock dropped 47.9% during sustained load (initial 1755 MHz → min 915 MHz, max_temp 59 °C). Thermal or power-limit throttling detected.
assert 0.47863247863247865 <= 0.1
- [failure] tests.test_system_health_and_reliability.TestQuantizationNumericalCorrectness::test_quantised_softmax_output_matches_fp16: AssertionError: Quantised softmax max error 2.56e-03 exceeds 1e-03
assert 0.0025634765625 < 0.001
- [failure] tests.test_system_health_and_reliability.TestTorchCompileCorrectnessAndSpeedup::test_compile_speedup_over_eager: AssertionError: torch.compile speedup 0.804× < floor 1.05×. Compiled kernels may be suboptimal for this H100 PCIe configuration.
assert 0.8038794604279713 >= 1.05

### performance_suite

- [failure] tests.test_performance_suite::test_performance_case_suite[AI_HOSTCPU_PERF_TEST_ID_011]: Failed: Command failed for AI_HOSTCPU_PERF_TEST_ID_011: genai-perf profile --endpoint-type chat --service-kind openai --url http://127.0.0.1:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct
executed_command:
genai-perf profile --endpoint-type chat --service-ki

### mlperf_suite

- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_003]: Failed: AI_HOSTCPU_ML_TEST_ID_003: command failed (rc=1)
executed_command: vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 --tensor-parallel-size 4 --max-model-len 32768 --gpu-memory-utilization 0.90 --port 8000
stderr_tail:
(APIServer pid=70758) Traceback (mo
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_004]: Failed: Timeout (>1800.0s) from pytest-timeout.
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_005]: Failed: AI_HOSTCPU_ML_TEST_ID_005: command failed (rc=1)
executed_command: vllm serve meta-llama/Llama-2-70b-chat-hf --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
stderr_tail:
(APIServer pid=73088) Tra
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
executed_command: /opt/venvs/torch-cu128/bin/python /root/workloads/mlcommons/inference/recommendation/dlrm_v2/pytorch/run.py --scenario Offline --dataset-path /mnt/datasets/criteo --model-path /mnt/data
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_012]: Failed: AI_HOSTCPU_ML_TEST_ID_012: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/text_to_image/run_local.sh pytorch gpu --scenario Offline --dataset-path /mnt/datasets/coco-sdxl --model-path /mnt/datasets/sdxl --output arti
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_013]: Failed: AI_HOSTCPU_ML_TEST_ID_013: command failed (rc=127)
executed_command: bash /root/workloads/mlcommons/inference/graph/R-GAT/run_local.sh pytorch gpu --scenario Offline --dataset-path /mnt/datasets/igb-het --model-path /mnt/datasets/rgat --output artifact
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_014]: Failed: AI_HOSTCPU_ML_TEST_ID_014: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/language_model/gpt-j/run_training.py --model-name EleutherAI/gpt-j-6b --dataset-path /mnt/datasets/cnn_dailymail --output-
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_015]: Failed: AI_HOSTCPU_ML_TEST_ID_015: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/language_model/pytorch/bert/run_squad.py --bert_model bert-large-uncased --do_train --train_file /mnt/datasets/squad/train
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_016]: Failed: AI_HOSTCPU_ML_TEST_ID_016: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/image_segmentation/pytorch/train.py --data-dir /mnt/datasets/kits19 --epochs 4000 --batch-size 2 --optimizer adam --learni
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_017]: Failed: AI_HOSTCPU_ML_TEST_ID_017: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/stable_diffusion/main.py --base /root/workloads/mlcommons/training/stable_diffusion/configs/train_01x04x08.yaml --actual_r
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_018]: Failed: AI_HOSTCPU_ML_TEST_ID_018: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/language_model/pytorch/bert/run_pretraining.py --bert_model bert-large-uncased --do_train --train_file /mnt/datasets/bert/
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_019]: Failed: AI_HOSTCPU_ML_TEST_ID_019: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/large_language_model/megatron-lm/pretrain_gpt.py --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --num-lay
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_020]: Failed: AI_HOSTCPU_ML_TEST_ID_020: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/llm_finetune/run_lora_finetune.py --model_name_or_path meta-llama/Llama-2-70b-hf --dataset_name scrolls --dataset_config g
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_021]: Failed: AI_HOSTCPU_ML_TEST_ID_021: command failed (rc=1)
executed_command: torchrun --nproc_per_node=4 /root/workloads/mlcommons/training/recommendation/dlrm_v2/pytorch/dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot=13-512-256-128 --arch-mlp-t
