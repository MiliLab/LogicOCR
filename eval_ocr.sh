MODEL_PATH="OpenGVLab/InternVL3-8B"
IMAGE_FOLDER="your saved image folder containing LogicOCR images"
JSON_FILE="path to LogicOCR.json"
NUM_WORKERS=8


# OCR
python infer_ocr/internvl_3.py \
  --model_path $MODEL_PATH \
  --image_folder $IMAGE_FOLDER \
  --json_file $JSON_FILE \
  --num_workers $NUM_WORKERS \
  --output_folder ./res_ocr \
  --verbose

# evaluate OCR performance
python eval_ocr_tools/eval_all.py \
  --input_file res_ocr/InternVL3-8B_ocr.json \
  --output_file res_ocr/InternVL3-8B_ocr_res.json \
  --num_workers $NUM_WORKERS

# feed the OCR results for text-only reasoning, pay attention to the path of json file
python infer_ocr/internvl_3_answer.py \
  --model_path $MODEL_PATH \
  --json_file ./res_ocr/InternVL3-8B_ocr.json \
  --output_folder ./res_ocr \
  --num_workers $NUM_WORKERS \
  --lmm_input_modal text \
  --verbose
