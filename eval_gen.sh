MODEL_PATH="your LMM path"
IMAGE_FOLDER="your saved image folder containing LogicOCR-Gen images"
JSON_FILE="path to LogicOCR_gen.json"

OUTPUT_FOLDER="./res"
NUM_WORKERS=8
LMM_INPUT_MODAL="image-text"  # "image-text": multimodal input, "text": text-only input


python infer_models/qwen2_5_vl.py \
  --model_path $MODEL_PATH \
  --image_folder $IMAGE_FOLDER \
  --json_file $JSON_FILE \
  --output_folder $OUTPUT_FOLDER \
  --num_workers $NUM_WORKERS \
  --lmm_input_modal $LMM_INPUT_MODAL \
  --verbose


##### example for Qwen2.5-VL-72B-Instruct, 
##### if you want to test direct answering, add '--answer_directly'
# python infer_models/qwen2_5_vl.py \
#   --model_path Qwen/Qwen2.5-VL-72B-Instruct \
#   --image_folder ./images \
#   --json_file LogicOCR_gen.json \
#   --output_folder ./res \
#   --num_workers 8 \
#   --lmm_input_modal "image-text" \
#   --verbose \
#   --auto_device

##### example for NVILA, 
# torchrun --nproc-per-node=$NUM_WORKERS \
#   infer_models/nvila.py \
#   --model_path $MODEL_PATH \
#   --image_folder $IMAGE_FOLDER \
#   --json_file $JSON_FILE \
#   --output_folder $OUTPUT_FOLDER \
#   --lmm_input_modal $LMM_INPUT_MODAL \
#   --verbose

##### example for QvQ-72B-Preview API
# python infer_models/qvq_api.py \
#   --api_key $your_api \
#   --image_folder $IMAGE_FOLDER \
#   --json_file $JSON_FILE \
#   --output_filename qvq_72b_preview \
#   --output_folder $OUTPUT_FOLDER \
#   --num_workers $NUM_WORKERS \
#   --verbose

##### example for o4-mini
# python infer_models/o4-mini.py \
#   --api_key $your_api \
#   --base_url $your_base_url \
#   --image_folder $IMAGE_FOLDER \
#   --json_file $JSON_FILE \
#   --output_filename o4-mini \
#   --output_folder $OUTPUT_FOLDER \
#   --num_workers $NUM_WORKERS \
#   --verbose