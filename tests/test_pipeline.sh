rm -rf playground

uv run python -m UTGenerator.run_localization \
    --dataset_split verified \
    --dataset_slice :2 \
    --output_folder results/test_localization \
    --file_level \
    --related_level \
    --fine_grain_line_level \
    --top_n 3 \
    --compress \
    --context_window 10 \
    --temperature 1 \
    --num_sample 4 \
    --model o3

uv run python -m UTGenerator.run_localization \
    --merge \
    --output_folder results/test_merge \
    --start_file results/test_localization/loc_outputs.jsonl \
    --num_samples 4

uv run python -m UTGenerator.run_testgen \
    --loc_file results/test_merge/loc_merged_0-1_outputs.jsonl \
    --output_folder results/test_gen/new_gen_testCase_t099_lm01 \
    --loc_interval --top_n=3 --context_window=10 \
    --max_samples 2  --cot --diff_format \
    --gen_and_process 
