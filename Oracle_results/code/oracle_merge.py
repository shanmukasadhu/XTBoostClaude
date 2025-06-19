import argparse
import json
from pathlib import Path
from datetime import datetime

# ========= Argument Parsing =========
parser = argparse.ArgumentParser(description="Oracle experiment analysis: compute upper-bound performance across agents.")
parser.add_argument('--verified_path', type=str, required=True, help='Path to verified results folder')
args = parser.parse_args()

# ========= Load and Validate Path =========
verified_path = Path(args.verified_path)
if not verified_path.exists():
    print(f"Error: Path does not exist: {verified_path}")
    exit(1)

print("Starting Oracle Experiment Analysis")
print("=" * 60)

# ========= Data Structures =========
all_instances_global = set()
all_resolved_global = set()
model_instance_data = {}
model_resolved_data = {}

# ========= Load Data from Agents =========
print("\nLoading data from agents...")
for model_dir in verified_path.iterdir():
    if not model_dir.is_dir():
        continue

    results_file = model_dir / "results" / "results.json"
    if not results_file.exists():
        print(f"  Skipping {model_dir.name} - no results.json found")
        continue

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"  Error loading {model_dir.name}: {e}")
        continue

    # Collect all task instances for this model
    model_instances = set()
    categories = ["no_generation", "generated", "with_logs", "install_fail", 
                  "reset_failed", "no_apply", "applied", "test_errored", 
                  "test_timeout", "resolved"]
    for category in categories:
        if category in results and isinstance(results[category], list):
            model_instances.update(results[category])

    resolved_instances = set(results.get("resolved", []))

    model_instance_data[model_dir.name] = len(model_instances)
    model_resolved_data[model_dir.name] = {
        "count": len(resolved_instances),
        "instances": resolved_instances
    }

    all_instances_global.update(model_instances)
    all_resolved_global.update(resolved_instances)

    print(f"  {model_dir.name}: {len(model_instances)} tested, {len(resolved_instances)} resolved")

# ========= Oracle Statistics =========
total_unique_instances = len(all_instances_global)
total_unique_resolved = len(all_resolved_global)
oracle_pass_rate = (total_unique_resolved / total_unique_instances * 100) if total_unique_instances > 0 else 0

print("\n" + "=" * 60)
print("ORACLE EXPERIMENT RESULTS")
print("=" * 60)
print("\nOverall Statistics:")
print(f"  - Total agents analyzed: {len(model_instance_data)}")
print(f"  - Total unique instances across all agents: {total_unique_instances}")
print(f"  - Total unique instances resolved by any agent: {total_unique_resolved}")
print(f"  - Oracle Pass Rate (Upper Bound): {oracle_pass_rate:.2f}%")

# ========= Save instance_analysis.json =========
instance_analysis = {
    "metadata": {
        "analysis_date": datetime.now().isoformat(),
        "total_agents": len(model_instance_data),
        "verified_path": str(verified_path)
    },
    "summary": {
        "total_unique_instances": total_unique_instances,
        "num_agents": len(model_instance_data),
        "all_agents_same_count": len(set(model_instance_data.values())) == 1,
        "min_instances_per_agent": min(model_instance_data.values()) if model_instance_data else 0,
        "max_instances_per_agent": max(model_instance_data.values()) if model_instance_data else 0,
        "avg_instances_per_agent": sum(model_instance_data.values()) / len(model_instance_data) if model_instance_data else 0
    },
    "instances_per_agent": model_instance_data,
    "sample_instances": sorted(list(all_instances_global))[:20]
}
with open("instance_analysis.json", "w") as f:
    json.dump(instance_analysis, f, indent=2)
print("Instance analysis saved to: instance_analysis.json")

# ========= Save oracle_results.json =========
oracle_results = {
    "metadata": {
        "analysis_date": datetime.now().isoformat(),
        "total_agents": len(model_resolved_data),
        "verified_path": str(verified_path)
    },
    "oracle_metrics": {
        "total_unique_instances": total_unique_instances,
        "total_unique_resolved": total_unique_resolved,
        "oracle_pass_rate": oracle_pass_rate,
        "oracle_pass_rate_formatted": f"{oracle_pass_rate:.2f}%"
    },
    "per_agent_resolved": {
        agent: data["count"]
        for agent, data in sorted(model_resolved_data.items(), key=lambda x: x[1]["count"], reverse=True)
    },
    "top_10_agents": [
        {
            "agent": agent,
            "resolved_count": data["count"],
            "total_instances": model_instance_data.get(agent, 0),
            "individual_pass_rate": (data["count"] / model_instance_data.get(agent, 1) * 100)
            if model_instance_data.get(agent, 0) > 0 else 0
        }
        for agent, data in sorted(model_resolved_data.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
    ],
    "resolved_instances_sample": sorted(list(all_resolved_global))[:20]
}
with open("oracle_results.json", "w") as f:
    json.dump(oracle_results, f, indent=2)
print("Oracle results saved to: oracle_results.json")

# ========= Difficulty Distribution =========
print("\nInstance Difficulty Distribution:")
instance_solver_count = {}
for data in model_resolved_data.values():
    for inst in data["instances"]:
        instance_solver_count[inst] = instance_solver_count.get(inst, 0) + 1

difficulty_dist = {
    "solved_by_1_agent": sum(1 for c in instance_solver_count.values() if c == 1),
    "solved_by_2-5_agents": sum(1 for c in instance_solver_count.values() if 2 <= c <= 5),
    "solved_by_6-10_agents": sum(1 for c in instance_solver_count.values() if 6 <= c <= 10),
    "solved_by_>10_agents": sum(1 for c in instance_solver_count.values() if c > 10)
}
for category, count in difficulty_dist.items():
    print(f"  - {category}: {count} instances")

never_solved = total_unique_instances - total_unique_resolved
print(f"  - Never solved: {never_solved} instances")

# ========= Final Summary =========
print("\nAnalysis complete.")
print(f"\nKey Finding: Oracle pass rate is {oracle_pass_rate:.2f}%")
print(f"This means combining all {len(model_instance_data)} agents could theoretically solve {total_unique_resolved} out of {total_unique_instances} instances.")
