# XTBoost

## Environment Setup

1. clone the repo
    ```bash
    git clone --recurse-submodules https://github.com/uiuc-kang-lab/XTBoost.git
    ```
2. `cp .env.example .env` and put your OpenAI API key in `.env`.
3. install `uv` if you did not install it before.
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
4. install python and packages via `uv sync`

## Generating test cases using UTGenerator

1. locate the places for adding test cases by:
    ```bash
    uv run python -m UTGenerator.run_localization \
        --dataset_split verified \ # for SWE-bench Verified
        --dataset_slice :2 \ # first two samples
        --output_folder results/test_localization \ 
        --file_level \
        --related_level \
        --fine_grain_line_level \
        --top_n 3 \
        --compress \
        --context_window 10 \
        --temperature 1 \ # for reasoning models
        --num_sample 4 \
        --model o3
    uv run python -m UTGenerator.run_localization \
        --merge \
        --output_folder results/test_merge \
        --start_file results/test_localization/loc_outputs.jsonl \
        --num_samples 4
    ```
2. test case generation script:
    ```
    uv run python -m UTGenerator.run_testgen \
        --loc_file results/test_merge/loc_merged_0-1_outputs.jsonl \
        --output_folder results/test_gen/new_gen_testCase_t099_lm01 \
        --loc_interval --top_n=3 --context_window=10 \
        --max_samples 2  --cot --diff_format \
        --gen_and_process 
    ```
