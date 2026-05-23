#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-0,1,2,3}"
VISIBLE_GPUS="${VISIBLE_GPUS:-$GPUS}"
DEVICE="${DEVICE:-$GPUS}"
RUN_NAME="${RUN_NAME:-yolo11m_baseline}"
PROJECT="${PROJECT:-runs/aefc_yolo11}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${PROJECT}" "${LOG_DIR}"

CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" nohup python tools/train_aefc.py \
  --cfg configs/train_aefc.yaml \
  --device "${DEVICE}" \
  --project "${PROJECT}" \
  --name "${RUN_NAME}" \
  --log-dir "${LOG_DIR}" \
  --log-interval 100 \
  > "${LOG_DIR}/${RUN_NAME}.nohup.out" 2>&1 &

echo "started pid=$!"
echo "nohup output: ${LOG_DIR}/${RUN_NAME}.nohup.out"
echo "training log: ${LOG_DIR}/${RUN_NAME}.log"
