import json
import logging
import re
import os

from datasets import load_dataset
from tqdm import tqdm

from UTGenerator.fl.FL import LLMFL
from UTGenerator.util.preprocess_data import filter_none_python
from UTGenerator.util.utils import load_json, load_jsonl
from UTGenerator.util.get_repo_structure import get_project_structure_from_scratch

def extractFileFromString(patch_text):
    # Regex pattern to find file paths from the diff --git line
    pattern = r'^diff --git a/(.+?) b/'

    # Find all matches for the pattern
    matches = re.findall(pattern, patch_text, re.MULTILINE)

    # Return the extracted file paths
    return matches

def localize(args):
    PROJECT_FILE_LOC = os.environ.get("PROJECT_FILE_LOC", None)

    if args.dataset_split == "lite":
        swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    elif args.dataset_split == "full":
        swe_bench_data = load_dataset("princeton-nlp/SWE-bench", split="test")
    elif args.dataset_split == "verified":
        swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        
    # slice the dataset if specified
    if args.dataset_slice != ":":
        start, end = args.dataset_slice.split(":")
        start = int(start) if start != "" else 0
        end = int(end) if end != "" else swe_bench_data.num_rows
        swe_bench_data = swe_bench_data.select(range(start, end))
    
    if args.start_file:
        start_file_locs = load_jsonl(args.start_file)

    for bug in tqdm(swe_bench_data):

        if args.target_id is not None:
            if args.target_id != bug["instance_id"]:
                continue

        if PROJECT_FILE_LOC is not None:
            project_file = os.path.join(PROJECT_FILE_LOC, bug["instance_id"] + ".json")
            d = load_json(project_file)
        else:
            # we need to get the project structure directly
            d = get_project_structure_from_scratch(
                bug["repo"], bug["base_commit"], bug["instance_id"], "playground"
            )

        instance_id = d["instance_id"]

        logging.info(f"================ localize {instance_id} ================")

        bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
        problem_statement = bench_data["problem_statement"]
        test_patch = bench_data["test_patch"]
        test_patch_files = extractFileFromString(test_patch)
        structure = d["structure"]
        filter_none_python(structure)

        found_files = []
        found_related_locs = []
        found_edit_locs = []

        additional_artifact_loc_file = None
        additional_artifact_loc_related = None
        additional_artifact_loc_edit_location = None
        file_traj, related_loc_traj, edit_loc_traj = {}, {}, {}

        # file level localization
        if args.file_level:
            fl = LLMFL(
                d["instance_id"],
                structure,
                problem_statement,
                test_patch,
                test_patch_files,
                args.model,
                args.backend,
            )
            found_files, additional_artifact_loc_file, file_traj = fl.localize(
                mock=args.mock
            )
        else:
            # assume start_file is provided
            for locs in start_file_locs:
                if locs["instance_id"] == d["instance_id"]:
                    found_files = locs["found_files"]
                    additional_artifact_loc_file = locs["additional_artifact_loc_file"]
                    file_traj = locs["file_traj"]

                    if "found_related_locs" in locs:
                        found_related_locs = locs["found_related_locs"]
                        additional_artifact_loc_related = locs[
                            "additional_artifact_loc_related"
                        ]
                        related_loc_traj = locs["related_loc_traj"]
                    break

        # related class, functions, global var localization
        if args.related_level:
            if len(found_files) != 0:
                pred_files = found_files[: args.top_n]
                fl = LLMFL(
                    d["instance_id"],
                    structure,
                    problem_statement,
                    test_patch,
                    test_patch_files,
                    args.model,
                    args.backend,
                )

                additional_artifact_loc_related = []
                found_related_locs = []
                related_loc_traj = {}

                if args.compress:
                    (
                        found_related_locs,
                        additional_artifact_loc_related,
                        related_loc_traj,
                    ) = fl.localize_function_from_compressed_files(
                        pred_files,
                        mock=args.mock,
                    )
                    additional_artifact_loc_related = [additional_artifact_loc_related]
                else:
                    assert False, "Not implemented yet."

        if args.fine_grain_line_level:
            # Only supports the following args for now

            pred_files = found_files[: args.top_n]
            fl = LLMFL(
                instance_id,
                structure,
                problem_statement,
                test_patch,
                test_patch_files,
                args.model,
                args.backend,
            )
            coarse_found_locs = {}
            for i, pred_file in enumerate(pred_files):
                if len(found_related_locs) > i:
                    coarse_found_locs[pred_file] = found_related_locs[i]
            (
                found_edit_locs,
                additional_artifact_loc_edit_location,
                edit_loc_traj,
            ) = fl.localize_line_from_coarse_function_locs(
                pred_files,
                coarse_found_locs,
                context_window=args.context_window,
                add_space=args.add_space,
                no_line_number=args.no_line_number,
                sticky_scroll=args.sticky_scroll,
                mock=args.mock,
                temperature=args.temperature,
                num_samples=args.num_samples,
            )

            additional_artifact_loc_edit_location = [
                additional_artifact_loc_edit_location
            ]

        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "instance_id": d["instance_id"],
                        "found_files": found_files,
                        "additional_artifact_loc_file": additional_artifact_loc_file,
                        "file_traj": file_traj,
                        "found_related_locs": found_related_locs,
                        "additional_artifact_loc_related": additional_artifact_loc_related,
                        "related_loc_traj": related_loc_traj,
                        "found_edit_locs": found_edit_locs,
                        "additional_artifact_loc_edit_location": additional_artifact_loc_edit_location,
                        "edit_loc_traj": edit_loc_traj,
                    }
                )
                + "\n"
            )
        break


def merge(args):
    """Merge predicted locations."""
    start_file_locs = load_jsonl(args.start_file)

    # Dump each location sample.
    for st_id in range(args.num_samples):
        en_id = st_id
        merged_locs = []
        for locs in start_file_locs:
            if "found_edit_locs" not in locs or len(locs["found_edit_locs"]) <= st_id:
                continue  

            merged_found_locs = [
                "\n".join(x) for x in locs["found_edit_locs"][st_id]
            ]
            merged_locs.append({**locs, "found_edit_locs": merged_found_locs})

        with open(
            f"{args.output_folder}/loc_merged_{st_id}-{en_id}_outputs.jsonl", "w"
        ) as f:
            for data in merged_locs:
                f.write(json.dumps(data) + "\n")


    # Pair wise merge
    for st_id in range(0, args.num_samples - 1, 2):
        en_id = st_id + 1
        print(f"Merging sample {st_id} and {en_id}...")
        merged_locs = []
        for locs in start_file_locs:
            if "found_edit_locs" not in locs or len(locs["found_edit_locs"]) <= en_id:
                continue

            try:
                merged_found_locs = ["\n".join(x) for x in locs["found_edit_locs"][st_id]]
                for sample_found_locs in locs["found_edit_locs"][st_id + 1 : en_id + 1]:
                    for i, file_found_locs in enumerate(sample_found_locs):
                        if isinstance(file_found_locs, str):
                            merged_found_locs[i] += "\n" + file_found_locs
                        else:
                            merged_found_locs[i] += "\n" + "\n".join(file_found_locs)
            except Exception as e:
                print(f"[Warning] Skipping {locs.get('instance_id')} pair {st_id}-{en_id} due to error: {e}")
                continue

            merged_locs.append({**locs, "found_edit_locs": merged_found_locs})

        with open(f"{args.output_folder}/loc_merged_{st_id}-{en_id}_outputs.jsonl", "w") as f:
            for data in merged_locs:
                f.write(json.dumps(data) + "\n")

    ### Merge all
    all_merged_locs = []
    print("Merging all samples...")
    for locs in start_file_locs:
        if "found_edit_locs" not in locs or len(locs["found_edit_locs"]) == 0:
            continue

        try:
            merged_found_locs = ["\n".join(x) for x in locs["found_edit_locs"][0]]
            for sample_found_locs in locs["found_edit_locs"][1:]:
                for i, file_found_locs in enumerate(sample_found_locs):
                    if isinstance(file_found_locs, str):
                        merged_found_locs[i] += "\n" + file_found_locs
                    else:
                        merged_found_locs[i] += "\n" + "\n".join(file_found_locs)
        except Exception as e:
            print(f"[Warning] Skipping {locs.get('instance_id')} during all-sample merge due to error: {e}")
            continue

        all_merged_locs.append({**locs, "found_edit_locs": merged_found_locs})

    with open(f"{args.output_folder}/loc_all_merged_outputs.jsonl", "w") as f:
        for data in all_merged_locs:
            f.write(json.dumps(data) + "\n")
