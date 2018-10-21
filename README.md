# kaggle-rsna

```
s3fs mirco-kaggle -o use_cache=/tmp -o allow_other -o uid=1000 -o mp_umask=002 -o multireq_max=5 /mirco-kaggle
```

```
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr
```

```
export CHECKPOINT_NUMBER=1000
```

```
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${MODEL_DIR}/model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory ${MODEL_EXP_DIR}
```

```
python object_detection/inference/infer_detections.py \
    --input_tfrecord_paths=${BASE_DIR}/data/test.tfrec \
    --output_tfrecord_path=${BASE_DIR}/pred/test.tfrec \
    --inference_graph=${MODEL_EXP_DIR}/frozen_inference_graph.pb \
    --discard_image_pixels
```
