import argparse
import logging
import json
import os
from dotenv import load_dotenv

from UTGenerator.fl.localize import merge, localize


load_dotenv()
# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
os.environ['OPENAI_API_KEY'] = api_key

# Load Anthropic API key (if exists)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if anthropic_api_key:
    os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_split", type=str, default="lite")
    parser.add_argument("--dataset_slice", type=str, default=":")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument(
        "--start_file",
        type=str,
        help="""previous output file to start with to reduce
        the work, should use in combination without --file_level""",
    )
    parser.add_argument("--file_level", action="store_true")
    parser.add_argument("--related_level", action="store_true")
    parser.add_argument("--fine_grain_line_level", action="store_true")
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--no_line_number", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument("--target_id", type=str)
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )
    parser.add_argument(
        "--model", type=str, default=os.getenv("OPENAI_MODEL_NAME")
    )
    parser.add_argument("--backend", type=str, default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--system_message", type=str, default="You are a helpful assistant.",
                        help="System message for the model, especially useful for Anthropic models")

    args = parser.parse_args()

    # Validate backend selection
    if args.backend == "anthropic":
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables but backend is set to 'anthropic'")
        # If user hasn't specified a model, set default Claude model
        if not args.model:
            args.model = "claude-3-sonnet-20240229"
            print(f"No model specified for Anthropic backend. Using default: {args.model}")
        # If user specified o3 or other OpenAI model, convert to Claude model
        elif args.model.startswith("o3") or args.model.startswith("gpt"):
            old_model = args.model
            args.model = "claude-3-sonnet-20240229"
            print(f"Converting OpenAI model '{old_model}' to Claude model: {args.model}")

    args.output_file = os.path.join(args.output_folder, args.output_file)

    assert not os.path.exists(args.output_file), "Output file already exists"

    assert not (
        args.file_level and args.start_file
    ), "Cannot use both file_level and start_file"

    assert not (
        args.file_level and args.fine_grain_line_level and not args.related_level
    ), "Cannot use both file_level and fine_grain_line_level without related_level"

    assert not (
        (not args.file_level) and (not args.start_file)
    ), "Must use either file_level or start_file"

    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    logging.basicConfig(
        filename=f"{args.output_folder}/localize.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if args.merge:
        merge(args)
    else:
        localize(args)


if __name__ == "__main__":
    main()
