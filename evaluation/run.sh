#!/bin/bash

DATA_PATH="../data/RadRevise_v0.csv"
BATCH_SIZE=32
OUTPUT_DIR="output/"

# List of model_ids
MODEL_IDS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "mistralai/MMistral-7B-Instruct-v0.3"
    "microsoft/Phi-3-mini-128k-instruct"
    "tiiuae/falcon-7b-instruct"
    "AdaptLLM/medicine-chat"
    "ruslanmv/Medical-Llama3-8B"
    "epfl-llm/meditron-7b"
)

mkdir -p "$OUTPUT_DIR"

for MODEL_ID in "${MODEL_IDS[@]}"
do
    OUTPUT_FILE="$OUTPUT_DIR/result_${MODEL_ID}.csv"
    python eval_model "$MODEL_ID" "$DATA_PATH" "$BATCH_SIZE" "$OUTPUT_FILE"
done
