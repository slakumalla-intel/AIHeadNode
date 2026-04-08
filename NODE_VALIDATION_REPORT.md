# AI Head Node — Node Validation Report

Generated: 2026-04-08 08:21:30
Pytest command line: -m node_validation -vv

## Highlights

- Total testcases captured: 36
- Passed: 28
- Failed: 8
- Skipped: 0
- Attention required: One or more testcase failures were detected.

## Execution Status Table

| Test | Purpose | Result | Status | Analysis |
|---|---|---|---|---|
| tests/test_node_validation.py::TestCPUGPULaunchAndSyncOverhead::test_kernel_launch_latency_stats | Measure kernel launch latency distribution (min/p50/p95/p99/max). | PASS (0.024s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestCPUGPULaunchAndSyncOverhead::test_synchronize_overhead_stats | Measure cudaDeviceSynchronize() overhead distribution. | PASS (0.001s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestCPUGPUPowerAndThrottling::test_sustained_gpu_power_stability | Run sustained GPU workload and verify stable power (low variance). | FAILED: AssertionError: Power variance 55.1 W is too high (expected < 50 W) assert np.float64(55.149026493875866) < 50 | FAIL | Check assertion thresholds and benchmark environment consistency. |
| tests/test_node_validation.py::TestGPUEnumerationAndTopology::test_gpu_enumeration | Enumerate all GPUs and verify count. | PASS (0.048s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestGPUEnumerationAndTopology::test_pcie_topology_discovery | TC-NV-01b: Validate GPU placement via nvidia-smi topo -m and print negotiated PCIe link speed/width. | PASS (0.634s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestGPUEnumerationAndTopology::test_topology_consistency | Validate GPU count and topology matrix consistency. | PASS (0.157s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestGPUP2PBandwidth::test_p2p_bandwidth_latency_binary | Run p2pBandwidthLatencyTest and validate bidirectional P2P BW ≥ threshold. | PASS (2.386s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestGPUP2PBandwidth::test_p2p_bandwidth_pytorch_fallback | PyTorch direct P2P measurement for all GPU pairs (sanity check). | FAILED: AssertionError: P2P BW GPU2->GPU3 (29.4 GB/s) below minimum 40.0 GB/s assert 29.36975753315087 >= 40.0 | FAIL | Check assertion thresholds and benchmark environment consistency. |
| tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_all_gather_perf | Run all_gather_perf and validate peak bus bandwidth. | PASS (3.399s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_all_reduce_perf | Run all_reduce_perf across all GPUs and validate peak busbw. | PASS (6.952s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_availability | Verify NCCL is accessible (via torch or test binary). | PASS (0.000s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_deterministic_init_variance | Run AllReduce 50× with small message and measure init/launch variance. | PASS (2.292s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_p2p_perf | Run p2p_perf (point-to-point send/recv) and validate bandwidth. | FAILED: AssertionError: NCCL sendrecv busbw 25.1 GB/s too low assert 25.05 >= 30.0 | FAIL | Check assertion thresholds and benchmark environment consistency. |
| tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nccl_reduce_scatter_perf | Run reduce_scatter_perf and validate peak bus bandwidth. | PASS (3.510s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestNCCLAndNVLinkValidation::test_nvlink_status | Verify NVLink is active and report per-link speed. | PASS (0.082s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestNVBandwidthMemcpy::test_device_to_device_memcpy_read_ce | CE D2D: device-to-device read bandwidth (HBM3 internal copy engine). | PASS (10.524s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestNVBandwidthMemcpy::test_device_to_host_bidirectional_memcpy_ce | CE bidirectional: D2H direction of simultaneous CE transfer. | PASS (9.760s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestNVBandwidthMemcpy::test_device_to_host_memcpy_ce | CE + pinned: D2H memcpy bandwidth via Copy Engine (all GPUs). | FAILED: AssertionError: device_to_host_memcpy_ce: min BW 39.8 GB/s below threshold 40.0 GB/s assert 39.81 >= 40.0 | FAIL | Check assertion thresholds and benchmark environment consistency. |
| tests/test_node_validation.py::TestNVBandwidthMemcpy::test_device_to_host_memcpy_sm | SM copy: D2H bandwidth using CUDA Streaming Multiprocessors (pageable). | PASS (5.128s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestNVBandwidthMemcpy::test_host_to_device_bidirectional_memcpy_ce | CE bidirectional: simultaneous H2D + D2H bandwidth via Copy Engine. | PASS (13.336s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestNVBandwidthMemcpy::test_host_to_device_memcpy_ce | CE + pinned: H2D memcpy bandwidth via Copy Engine (all GPUs). | PASS (4.708s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestNVBandwidthMemcpy::test_host_to_device_memcpy_sm | SM copy: H2D bandwidth using CUDA Streaming Multiprocessors (pageable). | PASS (4.970s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestPCIeBandwidthDirect::test_d2h_bandwidth_all_gpus | Measure D2H bandwidth for all GPUs and validate ≥ 65% of PCIe theoretical. | FAILED: AssertionError: GPU 1 D2H BW (33.4 GB/s) below threshold 41.0 GB/s assert 33.446584941509016 >= 40.95 | FAIL | Check assertion thresholds and benchmark environment consistency. |
| tests/test_node_validation.py::TestPCIeBandwidthDirect::test_h2d_bandwidth_all_gpus | Measure H2D bandwidth for all GPUs and validate ≥ 65% of PCIe theoretical (63 GB/s). | FAILED: AssertionError: GPU 3 H2D BW (33.8 GB/s) below threshold 41.0 GB/s assert 33.84259365348393 >= 40.95 | FAIL | Check assertion thresholds and benchmark environment consistency. |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_batch_throughput | Measure batched prefill throughput for HuggingFace GPT-2 (tokens/sec). | PASS (2.654s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency | Load a HuggingFace causal-LM (GPT-2) on GPU and measure per-token generation latency. | PASS (303.337s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[deepseek_r1_quantized-deepseek-ai/DeepSeek-R1-Distill-Llama-8B-32-3000.0-True] | Repeat latency validation for DeepSeek-R1 quantized, Llama, and Qwen workloads. | PASS (308.632s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[llama_70b-meta-llama/Meta-Llama-3-70B-16-12000.0-True] | Repeat latency validation for DeepSeek-R1 quantized, Llama, and Qwen workloads. | PASS (5310.154s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[llama_7b-meta-llama/Meta-Llama-3-8B-32-3000.0-False] | Repeat latency validation for DeepSeek-R1 quantized, Llama, and Qwen workloads. | PASS (305.092s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_hf_causal_lm_inference_latency_enterprise_workloads[qwen_70b-Qwen/Qwen2.5-72B-Instruct-16-12000.0-True] | Repeat latency validation for DeepSeek-R1 quantized, Llama, and Qwen workloads. | PASS (11550.456s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_hf_encoder_training_throughput | Fine-tune HuggingFace BERT for one epoch of synthetic data; validate throughput. | PASS (1.732s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_vision_image_generation_diffusers_throughput | TC-NV-08d: Vision workload (image generation) via Diffusers command.          Goal: validate stable image generation throughput.         Default command:           torchrun generate.py         Pass criteria:           - Throughput > 10 images/sec | PASS (0.007s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_vision_object_detection_yolo_fps | TC-NV-08c: Vision workload (object detection) via YOLO.          Goal: validate high aggregate FPS with efficient preprocessing.         Pass criteria:           - Aggregate FPS >= 200           - CPU utilization mean < 50% | FAILED: FileNotFoundError: [Errno 2] No such file or directory: 'yolo' | FAIL | Check assertion thresholds and benchmark environment consistency. |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_vllm_data_pipeline_saturation_inference_prefill | TC-NV-08b: Data pipeline saturation (inference prefill).          Goal: ensure GPUs do not idle due to CPU-side starvation while serving         prefill-heavy requests.          Workload command (vLLM benchmark):           python -m vllm.benchmarks.benchmark_serving --request-rate 100 --input-len-mean 1024          Pass criteria:           - GPU utilization sustained high (>= 90% for most samples)           - CPU utilization sustained below saturation (<= 70% for most samples) | FAILED: AssertionError: GPU utilization not sustained (>=90% ratio=0.00, mean=0.0%) assert 0.0 >= 0.8 | FAIL | Check assertion thresholds and benchmark environment consistency. |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_vllm_inference_throughput | Run vLLM offline inference with a HuggingFace model and measure throughput. | PASS (10.486s) | PASS | Executed successfully. |
| tests/test_node_validation.py::TestWorkloadAcceptanceHuggingFace::test_vllm_training_kpi_status | Capture vLLM training KPI status (vLLM is inference-focused and does not train). | PASS (0.000s) | PASS | Executed successfully. |
