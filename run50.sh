uv run python -m UTGenerator.run_localization \
  --dataset_split verified \
  --dataset_slice :85 \
  --output_folder results/test_localization_verified \
  --file_level \
  --related_level \
  --fine_grain_line_level \
  --top_n 3 \
  --compress \
  --context_window 10 \
  --temperature 1 \
  --num_samples 1 \
  --model claude-sonnet-4-20250514 \
  --backend anthropic

uv run python -m UTGenerator.run_localization \
  --dataset_split verified \
  --merge \
  --output_folder results/test_merge_verified \
  --start_file results/test_localization_verified/loc_outputs.jsonl \
  --num_samples 4

uv run python -m UTGenerator.run_testgen \
  --split verified \
  --loc_file results/test_merge_verified/loc_merged_0-1_outputs.jsonl \
  --output_folder results/test_gen_verified \
  --loc_interval \
  --top_n=3 \
  --context_window=10 \
  --max_samples 2 \
  --cot \
  --diff_format \
  --gen_and_process \
  --model claude-sonnet-4-20250514 \
  --backend anthropic \
  --max_tokens 4096


