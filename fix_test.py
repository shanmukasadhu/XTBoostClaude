import json
from datasets import load_dataset

# Step 1: Load SWE-bench Verified dataset
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

# Step 2: Load model output patches
run3_data = []
with open("/users/yuxuan18/codemonkeys-workdir/codemonkeys/codemonkeys_exampletest/codemonkeys_tests.jsonl", "r") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("model_patch", "").strip() != "":
            run3_data.append(entry)

# Convert model output into a dictionary for fast lookup
run3_map = {item["instance_id"]: item["model_patch"] for item in run3_data}

# Step 3: Replace patches in the SWE-bench dataset
modified_instances = []

for item in ds:
    instance_id = item["instance_id"]
    if instance_id in run3_map:
        new_patch = run3_map[instance_id]
        item["test_patch"] = new_patch
        item["patch"] = new_patch
        modified_instances.append(item)

# Step 4: Output the modified dataset to a JSON file
with open("codemonkey_tests.json", "w") as f:
    json.dump(modified_instances, f, indent=2)

print(f"Saved {len(modified_instances)} modified instances to final_modified_swebench.json")
