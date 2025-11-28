MODEL_PATH="your LMM path"
IMAGE_FOLDER="your saved image folder containing LogicOCR-Real images"
JSON_FILE="path to LogicOCR_real.json"

OUTPUT_FOLDER="./res_real"
NUM_WORKERS=8

python infer_models_real/qwen2_5_vl.py \
  --model_path $MODEL_PATH \
  --image_folder $IMAGE_FOLDER \
  --json_file $JSON_FILE \
  --output_folder $OUTPUT_FOLDER \
  --num_workers $NUM_WORKERS \
  --verbose