# Parallel Soak Test — Operator Guide

`run_parallel_soak_12h.sh` / `run_parallel_soak_24h.sh` drive four workloads in parallel to
stress a GPU node end-to-end:

1. **stress-ng** — CPU + memory pressure
2. **fio** — disk I/O pressure on a chosen mount
3. **Llama-3.1-8B fine-tune loop** — GPU load (FSDP-sharded, restarts for the full duration)
4. **Health snapshot monitor** — periodic `uptime`, `free`, `df`, `nvidia-smi`, `iostat`, `mpstat`

The two scripts are identical except for the default duration (12 h vs 24 h).

---

## 1. Prerequisites

Required commands (script exits if any are missing): `stress-ng`, `fio`, `timeout`.
Also used: `python3`, `pip`, `torchrun`, `curl`, `df`, `nvidia-smi`.

Python env must have a working `transformers`, `torch`, `datasets`, `accelerate`.
A valid **`HF_TOKEN`** is required (Llama-3.1 is a gated model).

```bash
# In the target venv on the node:
export HF_TOKEN=<your_token>        # do NOT echo the value into logs
python3 -c "import transformers, torch; print(transformers.__version__, torch.__version__)"
```

---

## 2. Pre-flight sanity checks

Run these on the node **before** any long soak. Paste one line at a time.

```bash
# a) Syntax
bash -n scripts/run_parallel_soak_12h.sh && echo "SYNTAX OK"

# b) Required tools present
for c in stress-ng fio timeout python3 pip torchrun curl df nvidia-smi; do command -v "$c" >/dev/null 2>&1 && echo "OK   $c" || echo "MISSING $c"; done

# c) Disk path is mounted, writable, has room
DISK=/mnt/nvme1n1p1; df -h "$DISK"; test -w "$DISK" && echo "writable OK" || echo "NOT WRITABLE"

# d) Memory headroom (see FSDP / stress-ng notes below)
free -h

# e) GPU count must match ML_NUM_GPUS (default 4)
nvidia-smi --query-gpu=index,name,memory.total --format=csv; echo "GPUs: $(nvidia-smi -L | wc -l)"

# f) Token / libs
echo "HF_TOKEN set? ${HF_TOKEN:+yes}${HF_TOKEN:-NO}"
python3 -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"
```

---

## 3. Quick smoke test (always do this first)

A 1-hour smoke validates every workload path without committing to a full run.

```bash
unset LLAMA_TRAIN_CMD          # let the script build its fixed default command
export ML_NUM_GPUS=$(nvidia-smi -L | wc -l)

./scripts/run_parallel_soak_12h.sh \
  --duration-hours 1 \
  --disk-path /mnt/nvme1n1p1 \
  --fio-size-gb 5 --fio-numjobs 2 --fio-iodepth 16 --fio-bs 1M \
  --snapshot-interval-sec 30 \
  --log-dir artifacts/soak_smoke_$(date +%Y%m%d_%H%M%S)
```

**Pass criteria:**
- Startup line shows `dataset: Salesforce/wikitext/wikitext-103-raw-v1, FSDP full_shard`
- `llama_training.log` shows `loss=` advancing past step 1 (no `OutOfMemoryError`)
- `stressng.log` and `fio.log` are actively updating
- `dmesg -T | grep -i oom` is empty

---

## 4. Full soak

```bash
unset LLAMA_TRAIN_CMD
export ML_NUM_GPUS=$(nvidia-smi -L | wc -l)

# 12-hour
./scripts/run_parallel_soak_12h.sh \
  --duration-hours 12 \
  --cpu-cores $(nproc) \
  --disk-path /mnt/nvme1n1p1 \
  --fio-size-gb auto \
  --snapshot-interval-sec 60 \
  --log-dir artifacts/soak_$(date +%Y%m%d_%H%M%S)

# 24-hour: same flags, use run_parallel_soak_24h.sh
```

Run detached so an SSH drop doesn't kill it:
```bash
nohup ./scripts/run_parallel_soak_12h.sh --duration-hours 12 --disk-path /mnt/nvme1n1p1 \
  > /tmp/soak_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 5. Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--duration-hours <N>` | 12 / 24 | Soak length |
| `--cpu-cores <N>` | 224 | stress-ng CPU workers (set to `$(nproc)` for full node) |
| `--total-mem-gb <N>` | 371 | Drives stress-ng memory (see note below) |
| `--disk-path <PATH>` | `/` | Where fio writes its test file |
| `--fio-size-gb <N\|auto>` | auto | fio file size (`auto` = 50% of free space, min 10 G) |
| `--fio-numjobs <N>` | 8 | fio parallel jobs |
| `--fio-iodepth <N>` | 64 | fio queue depth |
| `--fio-bs <SIZE>` | 1M | fio block size |
| `--snapshot-interval-sec <N>` | 60 | Health snapshot cadence |
| `--log-dir <PATH>` | `artifacts/soak_<ts>` | Output directory |
| `--llama-cmd <CMD>` | (built-in FSDP default) | Override the GPU workload entirely |

Useful env overrides: `HF_MODEL`, `HF_DATASET`, `HF_DATASET_CONFIG`, `ML_NUM_GPUS`,
`FSDP_CONFIG_FILE`, `HF_OUTPUT_DIR`.

---

## 6. Memory / GPU notes

**stress-ng memory:** the script sets `--vm-bytes` to `TOTAL_MEM_GB * 50%` across **8 workers**,
i.e. it requests roughly `TOTAL_MEM_GB * 4` in total. Ensure that fits in available RAM.
Example: `--total-mem-gb 371` → ~1.48 TB requested; fine on a 2.7 TiB-free box, but would OOM
a 371 GB box. Lower `--total-mem-gb` for smaller nodes.

**GPU memory (why FSDP):** full-parameter Adam training of an 8B model needs ~128 GB/GPU
(16 GB weights + 16 GB grads + ~96 GB optimizer). That exceeds an 80 GB H100, so plain DDP OOMs.
The default command uses **FSDP full_shard** to split params/grads/optimizer across all GPUs
(~30–40 GB/GPU) plus activation checkpointing. FSDP config is written to
`/tmp/hf_fsdp_config.json` at runtime.

---

## 7. Monitoring

```bash
LOG=$(ls -td artifacts/soak_* | head -1)
tail -f "$LOG"/llama_training.log "$LOG"/stressng.log "$LOG"/fio.log
watch -n5 nvidia-smi
grep -iE "oom|killed process" <(dmesg -T)
```

Output files per run (in `--log-dir`):
- `stressng.log`, `fio.log`, `llama_training.log`, `health_timeline.log`

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `torch.OutOfMemoryError` at step 1 | Plain DDP full fine-tune too big for 80 GB | Use the default FSDP command; if still OOM lower `--block_size` to 512 or `--optim adamw_bnb_8bit` |
| `HfUriError: Repository id must be 'namespace/name', got 'wikitext'` | New `datasets` rejects bare alias | Use `Salesforce/wikitext` (already the default) |
| `Some specified arguments are not used ... --fsdp_transformer_layer_cls_to_wrap` | transformers 5.x removed that CLI flag | Put `transformer_layer_cls_to_wrap` in `--fsdp_config` JSON (already handled) |
| `ERROR: Required command not found: <x>` | Missing tool | Install `stress-ng` / `fio` |
| fio "NOT WRITABLE" | Bad `--disk-path` | Point to a writable mount |
| Gated model 403 / auth error | `HF_TOKEN` unset or no model access | Set a valid token; request access to the Llama repo |
| Pure infra test, skip GPU | — | `--llama-cmd 'bash -lc "sleep 30"'` |

**Security:** never `echo "$HF_TOKEN"` — it lands in logs/history. If a token is exposed, rotate it
at <https://huggingface.co/settings/tokens>.

---

## 9. Cleanup

The script removes its fio test file on exit. To stop a detached run:
```bash
pkill -f run_parallel_soak
```
Child workloads (stress-ng, fio, torchrun) exit via the script's trap on the next cycle;
verify with `pgrep -af 'stress-ng|fio|torchrun'`.
