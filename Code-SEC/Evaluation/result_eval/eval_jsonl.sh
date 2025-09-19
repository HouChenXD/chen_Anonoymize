#!/bin/bash
OUTPUT_FILE="your_output_file.jsonl"  # Replace with your actual output file name

PREFIX_INPUT_FILE="${OUTPUT_FILE}"
PREFIX_OUTPUT_FILE="${OUTPUT_FILE%.*}_processed.jsonl"

PS_FILE="$PREFIX_OUTPUT_FILE"

python prefix.py "$PREFIX_INPUT_FILE" "$PREFIX_OUTPUT_FILE"
python filter.py "$PS_FILE"