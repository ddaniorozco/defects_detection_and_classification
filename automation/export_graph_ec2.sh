#!/usr/bin/env bash

# Activate tensorflow virtual environment on aws deep learning ami
echo " -> source activate tensorflow_p36"
source activate tensorflow_p36
# source activate tensorflow2_p36

# Check GPU is working:
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

export PYTHONUNBUFFERED="TRUE"

if [[ -z "${WORKSPACE}" ]]; then
  export WORKSPACE="$(pwd)/workspace"
fi

pushd ~/ml/models/research
export PYTHONPATH="$(pwd):$(pwd)/slim"
echo "PYTHONPATH: ${PYTHONPATH}"

export LOCAL_WORK_DIR="${WORKSPACE}/automation/data/"
echo "LOCAL_WORK_DIR: ${LOCAL_WORK_DIR}"

echo " -> rm -rf ${LOCAL_WORK_DIR}"
rm -rf "${LOCAL_WORK_DIR}"

echo "PIPELINE_CONFIG_PATH: ${PIPELINE_CONFIG_PATH}"
echo "TRAINED_CKPT_PREFIX: ${TRAINED_CKPT_PREFIX}"

# export
# From tensorflow/models/research/
export INPUT_TYPE=image_tensor
export LOCAL_EXPORT_DIR="${LOCAL_WORK_DIR}graph"

echo "Exporting inference graph..."
python3 object_detection/export_inference_graph.py \
    --input_type="${INPUT_TYPE}" \
    --pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
    --trained_checkpoint_prefix="${TRAINED_CKPT_PREFIX}" \
    --output_directory="${LOCAL_EXPORT_DIR}"

# Saving the path to the jenkins job of this training
echo "${RUN_DISPLAY_URL}" >> "${LOCAL_EXPORT_DIR}/jenkins_job_${BUILD_ID}.txt"

echo "Uploading inference graph to ${EXPORT_DIR}"
aws s3 cp "${LOCAL_EXPORT_DIR}" "${EXPORT_DIR}" --recursive --quiet

echo "Export completed"
