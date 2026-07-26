#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_DIR="${DATASET_DIR:-}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/paper1_deblurred}"
STEPS="${STEPS:-50}"
DEVICE="${DEVICE:-cuda}"
BETTER_START="${BETTER_START:-1}"
TILED="${TILED:-0}"
TILE_SIZE="${TILE_SIZE:-512}"
TILE_STRIDE="${TILE_STRIDE:-256}"

if [[ -z "$DATASET_DIR" ]]; then
  DATASET_DIR="$(find /kaggle/input -type d -name 'paper1.v5i.yolov12' | head -n 1 || true)"
fi

if [[ -z "$DATASET_DIR" || ! -d "$DATASET_DIR" ]]; then
  echo "DATASET_DIR not found. Set it, for example:"
  echo "DATASET_DIR=/kaggle/input/blur-dm-final-dts/paper1.v5i.yolov12 bash kaggle_deblur_yolo_dataset.sh"
  exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
  for candidate in \
    /kaggle/working/model.pth \
    /kaggle/working/ckp_deBlur_diff/model.pth \
    /kaggle/working/ckp_deBlur_diff/DeblurDiff/checkpoint/model.pth \
    /kaggle/input/*/model.pth \
    /kaggle/input/*/*/model.pth
  do
    if [[ -f "$candidate" ]]; then
      MODEL_PATH="$candidate"
      break
    fi
  done
fi

if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
  echo "MODEL_PATH not found. Set it to your model.pth path, for example:"
  echo "MODEL_PATH=/kaggle/input/deblurdiff-model/model.pth bash kaggle_deblur_yolo_dataset.sh"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Dataset: $DATASET_DIR"
echo "Model:   $MODEL_PATH"
echo "Output:  $OUTPUT_DIR"

for split in train valid test; do
  input_images="$DATASET_DIR/$split/images"
  output_images="$OUTPUT_DIR/$split/images"

  if [[ ! -d "$input_images" ]]; then
    echo "Skip $split: $input_images does not exist"
    continue
  fi

  mkdir -p "$output_images"

  args=(
    --model "$MODEL_PATH"
    --input "$input_images"
    --output "$output_images"
    --device "$DEVICE"
    --steps "$STEPS"
  )

  if [[ "$BETTER_START" == "1" ]]; then
    args+=(--better_start)
  fi

  if [[ "$TILED" == "1" ]]; then
    args+=(--tiled --tile_size "$TILE_SIZE" --tile_stride "$TILE_STRIDE")
  fi

  python -u inference.py "${args[@]}"

  if [[ -d "$DATASET_DIR/$split/labels" ]]; then
    mkdir -p "$OUTPUT_DIR/$split/labels"
    cp -R "$DATASET_DIR/$split/labels/." "$OUTPUT_DIR/$split/labels/"
  fi
done

python - "$DATASET_DIR/data.yaml" "$OUTPUT_DIR/data.yaml" "$OUTPUT_DIR" <<'PY'
import os
import sys
import yaml

src_yaml, dst_yaml, output_dir = sys.argv[1:]
data = {}
if os.path.exists(src_yaml):
    with open(src_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

data["path"] = output_dir
data["train"] = "train/images"
data["val"] = "valid/images"
if os.path.isdir(os.path.join(output_dir, "test", "images")):
    data["test"] = "test/images"

with open(dst_yaml, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
PY

echo "Done. Deblurred YOLO dataset is in: $OUTPUT_DIR"
echo "Use this data yaml for YOLO: $OUTPUT_DIR/data.yaml"
