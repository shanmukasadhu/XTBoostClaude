#!/usr/bin/env python3
"""
Pre-download and cache all repositories and commits from the SWE-bench dataset
"""

import os
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm

from UTGenerator.util.get_repo_structure import get_project_structure_from_scratch
from UTGenerator.util.preprocess_data import LOCAL_REPO_CACHE

def main():
    parser = argparse.ArgumentParser(description="Cache repositories and commits from the SWE-bench dataset")
    parser.add_argument("--dataset_split", type=str, default="lite", choices=["lite", "verified"],
                        help="SWE-bench dataset split, options: 'lite' or 'verified'")
    parser.add_argument("--cache_dir", type=str, default=LOCAL_REPO_CACHE,
                        help="Cache directory path")
    parser.add_argument("--playground", type=str, default="playground",
                        help="Temporary download directory")
    args = parser.parse_args()
    
    # Set cache directory environment variable
    os.environ["LOCAL_REPO_CACHE"] = args.cache_dir
    
    # Ensure cache directory exists
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    
    # Load SWE-bench dataset
    if args.dataset_split == "verified":
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    else:
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    # Collect all unique repository and commit combinations
    repo_commits = {}
    for item in dataset:
        repo_name = item["repo"]
        commit_id = item["base_commit"]
        instance_id = item["instance_id"]
        
        if repo_name not in repo_commits:
            repo_commits[repo_name] = set()
        repo_commits[repo_name].add((commit_id, instance_id))
    
    # Download and cache each commit for each repository
    for repo_name, commits in tqdm(repo_commits.items(), desc="Processing repositories"):
        print(f"Processing repository: {repo_name}")
        for commit_id, instance_id in tqdm(commits, desc=f"Processing commits for {repo_name}"):
            print(f"  Caching commit: {commit_id}")
            try:
                # Get and cache project structure
                get_project_structure_from_scratch(
                    repo_name, commit_id, instance_id, args.playground
                )
                print(f"  Successfully cached {repo_name}@{commit_id}")
            except Exception as e:
                print(f"  Error while caching {repo_name}@{commit_id}: {e}")
    
    print(f"Done! All repositories and commits have been cached to {args.cache_dir}")

if __name__ == "__main__":
    main() 