#!/usr/bin/env bash

# Check GPU is working:
# tf1
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# tf2
# tf.test.is_gpu_available()
# or
# tf.config.list_physical_devices('GPU')

export PYTHONUNBUFFERED="TRUE"

export WORKSPACE="/workspace"

#pushd ~/ml/models/research
#export PYTHONPATH="$(pwd):$(pwd)/slim"
#echo "PYTHONPATH: ${PYTHONPATH}"

export LOCAL_TRAIN_DIR="${WORKSPACE}/automation/data/"
echo "LOCAL_TRAIN_DIR: ${LOCAL_TRAIN_DIR}"

echo " -> rm -rf ${LOCAL_TRAIN_DIR}"
rm -rf "${LOCAL_TRAIN_DIR}"

echo " -> python3 ${WORKSPACE}/automation/object_detection_training_configurator.py"
python3 "${WORKSPACE}/automation/object_detection_training_configurator.py" --remote_config_path "${REMOTE_PIPELINE_CONFIG_PATH}" --local_train_dir "${LOCAL_TRAIN_DIR}"
export PIPELINE_CONFIG_PATH=$(cat "${LOCAL_TRAIN_DIR}local_pipeline_config_path_file_name")
echo "PIPELINE_CONFIG_PATH: ${PIPELINE_CONFIG_PATH}"

# Creating local model dir
export LOCAL_MODEL_DIR="${LOCAL_TRAIN_DIR}model/"
echo "Creating local model dir: ${LOCAL_MODEL_DIR}"
pushd "${LOCAL_TRAIN_DIR}"
mkdir model
popd

echo "Copying checkpoints from previous trainings: ${MODEL_DIR}"
aws s3 sync "${MODEL_DIR}" "${LOCAL_MODEL_DIR}" --quiet

# Saving the path to the jenkins job of this training
echo "${RUN_DISPLAY_URL}" >> "${LOCAL_MODEL_DIR}jenkins_job_${BUILD_ID}.txt"

# Starting the sync process to run every minute
source "${WORKSPACE}/automation/s3_sync.sh" "${LOCAL_MODEL_DIR}" "${MODEL_DIR}" 180 &

echo "Starting tensorboard..."
tensorboard "--logdir=${LOCAL_MODEL_DIR}" &

echo "Running training..."
python3 object_detection/model_main.py \
    --pipeline_config_path="${PIPELINE_CONFIG_PATH}" \
    --model_dir="${LOCAL_MODEL_DIR}" \
    --alsologtostderr

# To make the log also be saved to s3, might impact the network for long training in case of big files
# 2>&1 | tee -a "${LOCAL_MODEL_DIR}log.txt"

# Clean up
pkill -f tensorboard
touch "${LOCAL_MODEL_DIR}../stop_training.txt"

echo "Training finished!"
