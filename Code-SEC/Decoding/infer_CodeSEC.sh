
MODEL_NAME_OR_PATH="gemma/codegemma-7b"
MODE="complete"
INPUT_FILE="gemma7b_gak.jsonl"
STEER_STEP=1
PREFIX=0
LAM=3
entropy_thresh=0.1
OUTPUT_FILE="output/output_gemma7b/output_gak/unextracted_completed_gemma7b_greedy_cm7_with_prefix_gak_dy_st_${STEER_STEP}_pre${PREFIX}_lam_${LAM}_ent_${entropy_thresh}_ckpt_32.jsonl"

export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES="0" python Decoding/infer_CodeSEC.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --contra_model_name_or_path "gemma7b/lora/contra_gak/checkpoint-xx" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --mode "$MODE" \
    --temperature 0.3 \
    --top_p 0.95 \
    --top_k 10 \
    --max_new_tokens 100 \
    --use_chat_template \
    --do_sample False \
    --num_beams 1 \
    --batch_size 8 \
    --top_n 10 \
    --lam $LAM \
    --steer_step $STEER_STEP \
    --prefix_step $PREFIX \
    --entropy_thresh $entropy_thresh \
    --is_zero_reweight True \
    --is_entropy_filter True \
    --dynamic
