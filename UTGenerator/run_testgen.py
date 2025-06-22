import os
import argparse

from UTGenerator.augtest.genTest import repair, post_process_repair

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="lite")
    parser.add_argument("--loc_file", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--loc_interval", action="store_true")
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument(
        "--stop_at_n_unique_valid_samples",
        type=int,
        default=-1,
        help="Early stop when we get N unique valid samples, set to -1 if don't want to do early stopping.",
    )
    parser.add_argument("--gen_and_process", action="store_true")
    parser.add_argument("--max_samples", type=int, default=20, help="Sampling budget.")
    parser.add_argument(
        "--select_id",
        type=int,
        default=-1,
        help="Index the selected samples during post-processing.",
    )
    parser.add_argument(
        "--model", type=str, default="o3"
    )

    parser.add_argument("--backend", type=str, default="openai", choices=["openai"])
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument(
        "--only_correct", action="store_true"
    )  # only work on correct loc files (saves time)
    parser.add_argument("--post_process", action="store_true")
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--fine_grain_loc_only", action="store_true")
    parser.add_argument("--diff_format", action="store_true")
    parser.add_argument("--skip_greedy", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate. If None, use the default value for the model.",)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling. 0.0 means greedy sampling.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    args.output_file = os.path.join(args.output_folder, "output.jsonl")

    if args.post_process:
        args.raw_output_file = args.output_file
        if args.select_id == -1:
            args.output_file = args.raw_output_file.replace(
                ".jsonl", "_processed.jsonl"
            )
        else:
            args.output_file = args.raw_output_file.replace(
                ".jsonl", f"_{args.select_id}_processed.jsonl"
            )
        post_process_repair(args)
    elif args.gen_and_process:
        repair(args)
        args.raw_output_file = args.output_file
        for i in range(args.max_samples):
            args.output_file = args.raw_output_file.replace(
                ".jsonl", f"_{i}_processed.jsonl"
            )
            args.select_id = i
            post_process_repair(args)
    else:
        repair(args)


if __name__ == "__main__":
    main()
