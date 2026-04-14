#!/usr/bin/env bash
# =============================================================================
# download_mlperf_datasets.sh
#
# Downloads and prepares all datasets required for AI_HOSTCPU_ML_TEST_ID_001–021
# on a 4×H100 SXM5 system.
#
# Usage:
#   bash scripts/download_mlperf_datasets.sh [OPTIONS]
#
# Options:
#   --data-dir DIR    Root directory for dataset storage  (default: /data/mlperf)
#   --skip-large      Skip datasets > 100 GB (ImageNet, Criteo Terabyte, LAION)
#   --dry-run         Print commands without executing
#   --only IDs        Comma-separated test IDs whose datasets to download
#                     e.g. --only 006,009,015
#   --jobs N          Parallel download threads for aria2c  (default: 16)
#
# Approximate dataset sizes:
#   ShareGPT JSON          ~  600 MB   (IDs 003-005)
#   ImageNet-2012 val      ~   6.4 GB  (ID 006)
#   OpenImages v6 val      ~  30 GB    (ID 007)
#   KiTS19                 ~  50 GB    (IDs 008, 016)
#   LibriSpeech test-clean ~   346 MB  (IDs 009, 010)
#   Criteo Day 23          ~  15 GB    (ID 011 — inference subset)
#   Criteo Terabyte (full) ~ 1.0 TB    (ID 021 — training, requires --no-skip-large)
#   COCO 2014 5k captions  ~  400 MB   (ID 012)
#   IGB-Het                ~  50 GB    (ID 013)
#   CNN/DailyMail          ~  400 MB   (ID 014)
#   SQuAD v1.1             ~   30 MB   (ID 015)
#   LAION-400M (subset)    ~  80 GB    (ID 017, requires --no-skip-large)
#   Wikipedia + BooksCorps ~  21 GB    (IDs 015, 018)
#   C4 (text_document)     ~  750 GB   (ID 019, requires --no-skip-large)
#   SCROLLS govreport      ~  600 MB   (ID 020)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA_DIR="${MLPERF_DATA_DIR:-/data/mlperf}"
SKIP_LARGE=false
DRY_RUN=false
ONLY_IDS=""
JOBS=16

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)    DATA_DIR="$2";    shift 2 ;;
    --skip-large)  SKIP_LARGE=true;  shift   ;;
    --dry-run)     DRY_RUN=true;     shift   ;;
    --only)        ONLY_IDS="$2";    shift 2 ;;
    --jobs)        JOBS="$2";        shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $*"
  else
    echo "[RUN] $*"
    "$@"
  fi
}

should_run() {
  local test_id="$1"
  if [[ -z "$ONLY_IDS" ]]; then return 0; fi
  local normalized
  normalized=$(printf "%03d" "${test_id##*_}" 2>/dev/null || echo "$test_id")
  if echo "$ONLY_IDS" | tr ',' '\n' | grep -qiE "(AI_HOSTCPU_ML_TEST_ID_0*${normalized#0*}|0*${normalized#0*})"; then
    return 0
  fi
  return 1
}

# Use aria2c when available (much faster for large datasets)
download() {
  local url="$1"
  local dest="$2"
  if command -v aria2c &>/dev/null; then
    run aria2c -x "${JOBS}" -s "${JOBS}" -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url"
  else
    run wget -q --show-progress -c -O "$dest" "$url"
  fi
}

log()  { echo "$(date '+%Y-%m-%d %H:%M:%S')  [INFO]  $*"; }
warn() { echo "$(date '+%Y-%m-%d %H:%M:%S')  [WARN]  $*" >&2; }

run mkdir -p "$DATA_DIR"
log "Dataset root: $DATA_DIR"

# =============================================================================
# Shared datasets
# =============================================================================

# ---------------------------------------------------------------------------
# ShareGPT  – used by IDs 003-005 (Mixtral / Llama inference benchmarks)
# ---------------------------------------------------------------------------
if should_run "003" || should_run "004" || should_run "005"; then
  SHAREGPT_DIR="${DATA_DIR}/sharegpt"
  if [[ ! -f "${SHAREGPT_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json" ]]; then
    log "[003-005] Downloading ShareGPT V3 dataset..."
    run mkdir -p "$SHAREGPT_DIR"
    download \
      "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" \
      "${SHAREGPT_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
  else
    log "[003-005] ShareGPT already present – skipping"
  fi
fi

# =============================================================================
# Inference datasets (IDs 006-013)
# =============================================================================

# ---------------------------------------------------------------------------
# ID 006 – ImageNet-2012 validation set  (50 000 images, ~6.4 GB)
# ---------------------------------------------------------------------------
if should_run "006"; then
  IMAGENET_DIR="${DATA_DIR}/imagenet"
  if [[ ! -f "${IMAGENET_DIR}/val_map.txt" ]]; then
    log "[006] ImageNet-2012 validation set..."
    run mkdir -p "$IMAGENET_DIR"
    warn "[006] ImageNet requires manual registration at https://image-net.org/download.php"
    warn "[006] After download, extract ILSVRC2012_img_val.tar to ${IMAGENET_DIR} and run:"
    warn "[006]   python3 \${MLPERF_INFERENCE_DIR}/vision/classification_and_detection/tools/prepare_imagenet.py --imagenet-dir ${IMAGENET_DIR}"
    # Optionally download the validation labels automatically:
    run wget -q --show-progress -O "${IMAGENET_DIR}/val_map.txt" \
      "https://raw.githubusercontent.com/mlcommons/inference/master/vision/classification_and_detection/tools/val_map.txt" || true
  else
    log "[006] ImageNet already present – skipping"
  fi
fi

# ---------------------------------------------------------------------------
# ID 007 – OpenImages v6 validation  (~2 500 images subset for MLPerf, ~5 GB)
# ---------------------------------------------------------------------------
if should_run "007"; then
  OI_DIR="${DATA_DIR}/openimages"
  if [[ ! -d "${OI_DIR}/validation" ]]; then
    log "[007] Downloading OpenImages v6 MLPerf validation subset..."
    run mkdir -p "$OI_DIR"
    # MLCommons maintains a pre-selected validation subset script
    run python3 -c "
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'fiftyone'])
import fiftyone.zoo as foz
ds = foz.load_zoo_dataset('open-images-v6', split='validation',
    label_types=['detections'], max_samples=24781, seed=42,
    dataset_dir='${OI_DIR}')
print('OpenImages downloaded:', ds)
" || warn "[007] fiftyone download failed – use the MLCommons prepare_openimages.sh script instead"
  else
    log "[007] OpenImages already present – skipping"
  fi
fi

# ---------------------------------------------------------------------------
# ID 008 / 016 – KiTS19  (medical segmentation, ~50 GB)
# ---------------------------------------------------------------------------
if should_run "008" || should_run "016"; then
  KITS_DIR="${DATA_DIR}/kits19"
  if [[ ! -d "${KITS_DIR}/case_00000" ]]; then
    log "[008/016] Downloading KiTS19 dataset..."
    run mkdir -p "$KITS_DIR"
    # Official KiTS19 download script
    if command -v git &>/dev/null; then
      run git clone --depth 1 https://github.com/neheller/kits19.git "${DATA_DIR}/kits19_repo"
      run python3 "${DATA_DIR}/kits19_repo/starter_code/get_imaging.py" --dest "$KITS_DIR"
    else
      warn "[008] git not found; clone https://github.com/neheller/kits19 manually"
    fi
  else
    log "[008/016] KiTS19 already present – skipping"
  fi
fi

# ---------------------------------------------------------------------------
# IDs 009 / 010 – LibriSpeech test-clean  (~346 MB)
# ---------------------------------------------------------------------------
if should_run "009" || should_run "010"; then
  LIBRI_DIR="${DATA_DIR}/librispeech"
  if [[ ! -d "${LIBRI_DIR}/test-clean" ]]; then
    log "[009/010] Downloading LibriSpeech test-clean..."
    run mkdir -p "$LIBRI_DIR"
    download \
      "https://www.openslr.org/resources/12/test-clean.tar.gz" \
      "${LIBRI_DIR}/test-clean.tar.gz"
    run tar -xzf "${LIBRI_DIR}/test-clean.tar.gz" -C "$LIBRI_DIR"
    log "[009/010] Generating LibriSpeech WAV manifest for MLPerf RNN-T..."
    run python3 -c "
import os, json, glob, pathlib
manifest = []
for flac in sorted(glob.glob('${LIBRI_DIR}/LibriSpeech/test-clean/**/*.flac', recursive=True)):
    p = pathlib.Path(flac)
    transcript_path = p.parent / (p.stem + '.txt')
    # Trans files are per-speaker; parse from trans.txt
    trans_dir = p.parent
    for tf in trans_dir.glob('*.trans.txt'):
        for line in tf.read_text().splitlines():
            parts = line.split(' ', 1)
            if parts[0] == p.stem:
                manifest.append({'audio_filepath': flac, 'duration': -1, 'text': parts[1].lower()})
with open('${LIBRI_DIR}/librispeech-dev-clean-wav.json', 'w') as fh:
    json.dump(manifest, fh)
print(f'Manifest written: {len(manifest)} samples')
" || warn "[009] Manifest generation skipped – run manually if needed"
  else
    log "[009/010] LibriSpeech test-clean already present – skipping"
  fi
fi

# ---------------------------------------------------------------------------
# ID 011 – Criteo Day 23  (inference subset, ~15 GB)
# ---------------------------------------------------------------------------
if should_run "011"; then
  CRITEO_DIR="${DATA_DIR}/criteo"
  if [[ ! -f "${CRITEO_DIR}/day_23" ]]; then
    log "[011] Downloading Criteo Terabyte Day 23 (inference subset, ~15 GB)..."
    run mkdir -p "$CRITEO_DIR"
    # The official Criteo Terabyte dataset is hosted at:
    download \
      "https://storage.googleapis.com/criteo-cail-datasets/day_23.gz" \
      "${CRITEO_DIR}/day_23.gz" || \
      warn "[011] Criteo Day 23 download failed; register at https://labs.criteo.com/2013/12/download-terabyte-click-logs/"
    if [[ -f "${CRITEO_DIR}/day_23.gz" ]]; then
      run gunzip -k "${CRITEO_DIR}/day_23.gz"
    fi
  else
    log "[011] Criteo Day 23 already present – skipping"
  fi
fi

# ---------------------------------------------------------------------------
# ID 012 – COCO 2014 captions  (5k evaluation subset for SDXL, ~400 MB)
# ---------------------------------------------------------------------------
if should_run "012"; then
  COCO_DIR="${DATA_DIR}/coco-sdxl"
  if [[ ! -f "${COCO_DIR}/captions_val2014.json" ]]; then
    log "[012] Downloading COCO 2014 captions for SDXL evaluation..."
    run mkdir -p "$COCO_DIR"
    download \
      "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" \
      "${COCO_DIR}/annotations.zip"
    run unzip -q "${COCO_DIR}/annotations.zip" -d "$COCO_DIR"
    # Download 5k validation images
    download \
      "http://images.cocodataset.org/zips/val2014.zip" \
      "${COCO_DIR}/val2014.zip"
    run unzip -q "${COCO_DIR}/val2014.zip" -d "$COCO_DIR"
    log "[012] Selecting MLPerf SDXL 5k image subset..."
    run python3 -c "
import json, shutil, pathlib, random
random.seed(2024)
ann = json.loads(pathlib.Path('${COCO_DIR}/annotations/captions_val2014.json').read_bytes())
imgs = random.sample(ann['images'], 5000)
out = {'info': ann.get('info',{}), 'images': imgs, 'annotations': [], 'licenses': []}
for a in ann['annotations']:
    if any(a['image_id'] == im['id'] for im in imgs):
        out['annotations'].append(a)
pathlib.Path('${COCO_DIR}/captions_val2014.json').write_text(json.dumps(out))
print('Wrote 5k COCO captions subset')
" || warn "[012] COCO subset selection failed – run manually"
  else
    log "[012] COCO-SDXL already present – skipping"
  fi
fi

# ---------------------------------------------------------------------------
# ID 013 – IGB-Het  (graph dataset, ~50 GB)
# ---------------------------------------------------------------------------
if should_run "013"; then
  IGB_DIR="${DATA_DIR}/igb-het"
  if [[ ! -d "$IGB_DIR" ]]; then
    log "[013] Downloading IGB-Het (Heterogeneous) graph dataset..."
    run mkdir -p "$IGB_DIR"
    run python3 -c "
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'igb'])
from igb.download import download_igb_het
download_igb_het(root='${IGB_DIR}', size='medium')
" || warn "[013] IGB-Het download failed – see https://github.com/IllinoisGraphBenchmark/IGB-Datasets"
  else
    log "[013] IGB-Het already present – skipping"
  fi
fi

# =============================================================================
# Training datasets (IDs 014-021)
# =============================================================================

# ---------------------------------------------------------------------------
# ID 014 – CNN/DailyMail  (GPT-J summarization)
# ---------------------------------------------------------------------------
if should_run "014"; then
  CNN_DIR="${DATA_DIR}/cnn_dailymail"
  if [[ ! -d "$CNN_DIR" ]]; then
    log "[014] Downloading CNN/DailyMail via datasets library..."
    run python3 -c "
from datasets import load_dataset
ds = load_dataset('cnn_dailymail', '3.0.0')
ds.save_to_disk('${CNN_DIR}')
print('CNN/DailyMail saved to ${CNN_DIR}')
" || warn "[014] CNN/DailyMail download failed; run: python3 -c \"from datasets import load_dataset; load_dataset('cnn_dailymail','3.0.0').save_to_disk('${CNN_DIR}')\""
  else
    log "[014] CNN/DailyMail already present – skipping"
  fi
fi

# ---------------------------------------------------------------------------
# ID 015 – SQuAD v1.1  (BERT SQuAD training)
# ---------------------------------------------------------------------------
if should_run "015"; then
  SQUAD_DIR="${DATA_DIR}/squad"
  if [[ ! -f "${SQUAD_DIR}/train-v1.1.json" ]]; then
    log "[015] Downloading SQuAD v1.1..."
    run mkdir -p "$SQUAD_DIR"
    download \
      "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json" \
      "${SQUAD_DIR}/train-v1.1.json"
    download \
      "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json" \
      "${SQUAD_DIR}/dev-v1.1.json"
  else
    log "[015] SQuAD v1.1 already present – skipping"
  fi
fi

# ---------------------------------------------------------------------------
# IDs 015 / 018 – Wikipedia + BooksCorpus pre-tokenised HDF5 for BERT pre-training
# ---------------------------------------------------------------------------
if should_run "015" || should_run "018"; then
  BERT_DATA_DIR="${DATA_DIR}/bert"
  if [[ ! -d "${BERT_DATA_DIR}/hdf5_lower_case_1_seq_len_128" ]]; then
    log "[015/018] Downloading BERT pre-training HDF5 dataset from MLCommons..."
    run mkdir -p "$BERT_DATA_DIR"
    # The MLCommons reference implementation provides a convenience download script
    warn "[015/018] BERT HDF5 download requires ~21 GB. Starting download..."
    if command -v gsutil &>/dev/null; then
      run gsutil -m cp -r \
        "gs://mlcommons-training-wg-public/bert_reference_results_text_md5/results4/seq_len_128" \
        "${BERT_DATA_DIR}/hdf5_lower_case_1_seq_len_128" || \
        warn "[015/018] gsutil download failed – see https://github.com/mlcommons/training/tree/master/language_model/pytorch/bert"
      run gsutil -m cp -r \
        "gs://mlcommons-training-wg-public/bert_reference_results_text_md5/results4/seq_len_512" \
        "${BERT_DATA_DIR}/hdf5_lower_case_1_seq_len_512" || true
    else
      warn "[015/018] gsutil not found. Install via: pip install google-cloud-storage"
      warn "[015/018] Then run: gsutil -m cp -r gs://mlcommons-training-wg-public/bert_reference_results_text_md5/results4/ ${BERT_DATA_DIR}/"
    fi
  else
    log "[015/018] BERT HDF5 already present – skipping"
  fi
fi

# ---------------------------------------------------------------------------
# ID 017 – LAION-400M webdataset subset  (~80 GB compressed)
# ---------------------------------------------------------------------------
if should_run "017" && [[ "$SKIP_LARGE" == "false" ]]; then
  LAION_DIR="${DATA_DIR}/laion400m"
  if [[ ! -d "$LAION_DIR" ]]; then
    log "[017] Downloading LAION-400M subset for SD training (~80 GB)..."
    run mkdir -p "$LAION_DIR"
    # img2dataset can efficiently download and convert LAION metadata
    run pip install -q img2dataset
    run python3 -c "
import img2dataset
img2dataset.download(
    url_list='https://huggingface.co/datasets/laion/laion400m/resolve/main/part-00000-5b54c5d5.parquet',
    image_size=512,
    output_type='webdataset',
    output_folder='${LAION_DIR}',
    processes_count=8,
    resize_mode='keep_ratio',
    min_image_size=256,
    max_image_area=4e6,
    number_sample_per_shard=10000,
)
" || warn "[017] LAION download failed – see https://github.com/rom1504/img2dataset"
  else
    log "[017] LAION-400M already present – skipping"
  fi
else
  log "[017] Skipping LAION-400M (--skip-large or not selected)"
fi

# ---------------------------------------------------------------------------
# ID 019 – C4 dataset  (GPT-3 pre-training, ~750 GB; skip by default)
# ---------------------------------------------------------------------------
if should_run "019" && [[ "$SKIP_LARGE" == "false" ]]; then
  C4_DIR="${DATA_DIR}/c4"
  if [[ ! -f "${C4_DIR}/c4_text_document_idx.npy" ]]; then
    log "[019] Downloading C4 and preparing Megatron-LM format (~750 GB)..."
    run mkdir -p "$C4_DIR"
    run python3 -c "
from datasets import load_dataset
ds = load_dataset('c4', 'en', split='train', streaming=False, cache_dir='${C4_DIR}/hf_cache')
ds.to_json('${C4_DIR}/c4_train.jsonl', lines=True, force_ascii=False)
print('C4 JSONL written – now run Megatron preprocess_data.py to produce binary')
"
    warn "[019] Next: run Megatron-LM tools/preprocess_data.py to produce c4_text_document.bin/.idx"
  else
    log "[019] C4 already present – skipping"
  fi
else
  log "[019] Skipping C4 (~750 GB) – pass without --skip-large to enable"
fi

# ---------------------------------------------------------------------------
# ID 020 – SCROLLS govreport  (Llama 2 LoRA fine-tuning)
# ---------------------------------------------------------------------------
if should_run "020"; then
  SCROLLS_DIR="${DATA_DIR}/scrolls"
  if [[ ! -d "$SCROLLS_DIR" ]]; then
    log "[020] Downloading SCROLLS govreport split..."
    run python3 -c "
from datasets import load_dataset
ds = load_dataset('tau/scrolls', 'gov_report')
ds.save_to_disk('${SCROLLS_DIR}')
print('SCROLLS govreport saved to ${SCROLLS_DIR}')
" || warn "[020] SCROLLS download failed; run: python3 -c \"from datasets import load_dataset; load_dataset('tau/scrolls','gov_report').save_to_disk('${SCROLLS_DIR}')\""
  else
    log "[020] SCROLLS already present – skipping"
  fi
fi

# ---------------------------------------------------------------------------
# ID 021 – Criteo Terabyte  (DLRMv2 training; ~1 TB; skip by default)
# ---------------------------------------------------------------------------
if should_run "021" && [[ "$SKIP_LARGE" == "false" ]]; then
  CRITEO_TRAIN_DIR="${DATA_DIR}/criteo"
  if [[ ! -f "${CRITEO_TRAIN_DIR}/terabyte_processed.npz" ]]; then
    log "[021] Criteo Terabyte full training set (~1 TB)..."
    run mkdir -p "$CRITEO_TRAIN_DIR"
    warn "[021] Registering/downloading Criteo Terabyte from https://labs.criteo.com/2013/12/download-terabyte-click-logs/"
    warn "[021] Each day_* file is ~15 GB; total 24 days = ~365 GB raw"
    for day in $(seq 0 23); do
      if [[ ! -f "${CRITEO_TRAIN_DIR}/day_${day}" ]]; then
        download \
          "https://storage.googleapis.com/criteo-cail-datasets/day_${day}.gz" \
          "${CRITEO_TRAIN_DIR}/day_${day}.gz" || \
          warn "[021] Failed to download day_${day}"
        [[ -f "${CRITEO_TRAIN_DIR}/day_${day}.gz" ]] && \
          run gunzip -k "${CRITEO_TRAIN_DIR}/day_${day}.gz"
      fi
    done
    log "[021] Pre-processing Criteo data with the MLCommons script..."
    warn "[021] Run: python3 \${MLPERF_TRAINING_DIR}/recommendation/dlrm_v2/pytorch/preprocess.py --input ${CRITEO_TRAIN_DIR} --output ${CRITEO_TRAIN_DIR}/terabyte_processed.npz"
  else
    log "[021] Criteo Terabyte already pre-processed – skipping"
  fi
else
  log "[021] Skipping Criteo Terabyte (~1 TB) – pass without --skip-large to enable"
fi

# =============================================================================
# Summary
# =============================================================================
log ""
log "=== Dataset download complete ==="
log "Data directory: $DATA_DIR"
log ""
if [[ "$DRY_RUN" == "false" ]]; then
  log "Disk usage:"
  du -sh "${DATA_DIR}/"* 2>/dev/null | sort -h || true
fi
log ""
log "Next step: bash scripts/run_mlperf_h100x4.sh"
