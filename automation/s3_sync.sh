#!/usr/bin/env bash

LOCAL_MODEL_DIR="$1"
REMOTE_MODEL_DIR="$2"
INTERVAL_SECONDS="$3"

echo "Saving checkpoints every ${INTERVAL_SECONDS} seconds"
echo "LOCAL_MODEL_DIR: ${LOCAL_MODEL_DIR}"
echo "REMOTE_MODEL_DIR: ${REMOTE_MODEL_DIR}"

while [ ! -f "${LOCAL_MODEL_DIR}../stop_training.txt" ];
do
  sleep "${INTERVAL_SECONDS}"
  echo "Saving checkpoints to ${REMOTE_MODEL_DIR}"
  aws s3 sync "${LOCAL_MODEL_DIR}" "${REMOTE_MODEL_DIR}" --delete --quiet
done

echo "S3 synchronization stopped"
