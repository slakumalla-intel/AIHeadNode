# Workload Execution Status Report

Generated (UTC): 2026-11-05T17:57:53.520713+00:00
Host: dmr-bkc

## Summary

- Domains run: 13
- Passed: 9
- Failed: 4
- Total tests: 411
- Failures: 28
- Errors: 0
- Skipped: 36

## Domain Status

| Domain | Status | Tests | Failures | Errors | Skipped | Duration (s) | XML | Log |
|---|---|---:|---:|---:|---:|---:|---|---|
| bottleneck | PASS | 25 | 0 | 0 | 0 | 34.529 | artifacts/reports/junit/bottleneck.xml | artifacts/reports/logs/bottleneck.log |
| cpu | PASS | 33 | 0 | 0 | 1 | 527.712 | artifacts/reports/junit/cpu.xml | artifacts/reports/logs/cpu.log |
| memory | FAIL | 28 | 1 | 0 | 0 | 23.334 | artifacts/reports/junit/memory.xml | artifacts/reports/logs/memory.log |
| gpu | PASS | 53 | 0 | 0 | 0 | 24.697 | artifacts/reports/junit/gpu.xml | artifacts/reports/logs/gpu.log |
| interconnect | FAIL | 77 | 2 | 0 | 8 | 78.011 | artifacts/reports/junit/interconnect.xml | artifacts/reports/logs/interconnect.log |
| node_validation | FAIL | 51 | 6 | 0 | 16 | 1278.548 | artifacts/reports/junit/node_validation.xml | artifacts/reports/logs/node_validation.log |
| agentic_ai | PASS | 9 | 0 | 0 | 0 | 20.534 | artifacts/reports/junit/agentic_ai.xml | artifacts/reports/logs/agentic_ai.log |
| ai_pipeline | PASS | 38 | 0 | 0 | 4 | 63.053 | artifacts/reports/junit/ai_pipeline.xml | artifacts/reports/logs/ai_pipeline.log |
| system_health | PASS | 39 | 0 | 0 | 2 | 57.995 | artifacts/reports/junit/system_health.xml | artifacts/reports/logs/system_health.log |
| system_config | PASS | 8 | 0 | 0 | 0 | 8.684 | artifacts/reports/junit/system_config.xml | artifacts/reports/logs/system_config.log |
| vllm_prefill | PASS | 3 | 0 | 0 | 1 | 199.823 | artifacts/reports/junit/vllm_prefill.xml | artifacts/reports/logs/vllm_prefill.log |
| performance_suite | PASS | 26 | 0 | 0 | 2 | 1551.549 | artifacts/reports/junit/performance_suite.xml | artifacts/reports/logs/performance_suite.log |
| mlperf_suite | FAIL | 21 | 19 | 0 | 2 | 6293.486 | artifacts/reports/junit/mlperf_suite.xml | artifacts/reports/logs/mlperf_suite.log |

## Failure Details

### memory

- [failure] tests.test_memory_validation.TestSingleThreadBandwidth::test_triad_bw_exceeds_15gbs: AssertionError: STREAM Triad 11.45 GB/s < expected 14.60 GB/s (40% of Copy 36.50 GB/s, floor 12 GB/s)
assert 11.454824424373532 >= 14.598929664283672

### interconnect

- [failure] tests.test_interconnect_validation.TestAllGPUH2D::test_concurrent_h2d_all_gpus: AssertionError: Concurrent H2D BW all 4 GPUs: efficiency 55.01% is below threshold 65.00% (measured=138.628, theoretical=252.000)
assert 0.5501129927213975 >= 0.65
- [failure] tests.test_interconnect_validation.TestNVLinkDetection::test_p2p_access_enabled: AssertionError: GPU 0 cannot access GPU 2 via P2P
assert False

### node_validation

- [failure] tests.test_node_validation.TestNCCLAndNVLinkValidation::test_nccl_p2p_perf: AssertionError: NCCL sendrecv busbw 20.2 GB/s too low
assert 20.24 >= 24.0
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[llama_70b-meta-llama/Meta-Llama-3.1-70B-Instruct-16-12000.0-True]: ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vllm_data_pipeline_saturation_inference_prefill: AssertionError: GPU utilization not sustained (>=90% ratio=0.00, mean=0.0%, target_ratio=0.50, shared_gpu=True)
assert 0.0 >= 0.5
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_vision_object_detection_yolo_fps: AssertionError: YOLO aggregate FPS too low: 170.5 < target 200.0
assert 170.47288547814924 >= 200.0
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_predictable_memory_orchestration_kv_cache_offload: AssertionError: Offload throughput degradation too high: 87.9% (baseline 394.2 -> offload 47.5)
assert 87.93874015647138 <= 60
- [failure] tests.test_node_validation.TestWorkloadAcceptanceHuggingFace::test_llm_training_small_multigpu: AssertionError: TC-013: gradient sync ratio 15.3% exceeds threshold 15% — communication bottleneck detected; check NVLink status and NCCL topology
assert 0.15295 <= 0.15

### mlperf_suite

- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_003]: Failed: Timeout (>1800.0s) from pytest-timeout.
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_004]: Failed: Timeout (>1800.0s) from pytest-timeout.
- [failure] tests.test_mlperf_suite::test_mlperf_case_suite[AI_HOSTCPU_ML_TEST_ID_005]: Failed: AI_HOSTCPU_ML_TEST_ID_005: command failed (rc=1)
executed_command: vllm serve meta-llama/Llama-2-70b-chat-hf --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
stderr_tail:
(APIServer pid=78963) Tra
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
executed_command: /mnt/nvme1n1p1/venvs/.venv-vllm-updated/bin/python /root/workloads/mlcommons/inference/recommendation/dlrm_v2/pytorch/run.py --scenario Offline --dataset-path /mnt/datasets/criteo --mod
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
