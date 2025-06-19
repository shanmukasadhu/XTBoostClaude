import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ========== Argument Parsing ==========
parser = argparse.ArgumentParser(description="Analyze cumulative oracle performance.")
parser.add_argument('--verified_path', type=str, required=True, help='Path to verified results folder')
args = parser.parse_args()

# ========== Load Verified Path ==========
verified_path = Path(args.verified_path)

if not verified_path.exists():
    print(f"Error: Path does not exist: {verified_path}")
    exit(1)

print("Cumulative Oracle Analysis - How Performance Grows with More Agents")
print("=" * 70)

# ========== Step 1: Load Agent Data ==========
agent_resolved_data = {}
agent_resolved_counts = {}
total_instances = 500  # Assumes SWE-bench Verified set

print("Loading agent data...")
for model_dir in verified_path.iterdir():
    if not model_dir.is_dir():
        continue

    results_file = model_dir / "results" / "results.json"
    if not results_file.exists():
        continue

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading {model_dir.name}: {e}")
        continue

    resolved_instances = set(results.get("resolved", []))
    if resolved_instances:
        agent_resolved_data[model_dir.name] = resolved_instances
        agent_resolved_counts[model_dir.name] = len(resolved_instances)
        print(f"{model_dir.name}: {len(resolved_instances)} resolved")

# ========== Step 2: Sort Agents by Performance ==========
sorted_agents = sorted(agent_resolved_counts.items(), key=lambda x: x[1], reverse=True)
print(f"\nFound {len(sorted_agents)} agents that resolved at least one instance")

# ========== Step 3: Calculate Cumulative Oracle Performance ==========
cumulative_resolved = []
cumulative_counts = []
agent_names = []
current_union = set()

print("Calculating cumulative oracle performance...")
for agent, count in sorted_agents:
    current_union.update(agent_resolved_data[agent])
    cumulative_resolved.append(len(current_union))
    cumulative_counts.append(count)
    agent_names.append(agent)

# ========== Step 4: Plot ==========
plt.figure(figsize=(12, 8))
x = range(1, len(cumulative_resolved) + 1)
y = cumulative_resolved

plt.plot(x, y, 'b-', linewidth=2, marker='o', markersize=4, label='Cumulative Unique Resolved')
plt.axhline(y=total_instances, color='r', linestyle='--', alpha=0.7, label=f'Total Instances ({total_instances})')

for percentage in [10, 20, 30, 40, 50]:
    y_val = total_instances * percentage / 100
    plt.axhline(y=y_val, color='gray', linestyle=':', alpha=0.3)

plt.xlabel('Number of Top Agents Combined', fontsize=12)
plt.ylabel('Unique Instances Resolved', fontsize=12)
plt.title('Oracle Performance Growth: Cumulative Unique Instances Resolved\nby Combining Top-Performing Agents', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.xlim(0, len(x) + 1)
plt.ylim(0, max(total_instances * 1.1, max(cumulative_resolved) * 1.1))

plt.tight_layout()
plt.savefig('oracle_growth_curve.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'oracle_growth_curve.png'")

# ========== Step 5: Report Statistics ==========
final_pass_rate = cumulative_resolved[-1] / total_instances * 100
print("\nKey Statistics:")
print(f"  - Final Oracle: {cumulative_resolved[-1]}/{total_instances} ({final_pass_rate:.1f}% pass rate)")
print(f"  - Top agent alone: {cumulative_counts[0]} instances")
print("  - Number of agents for 75% of oracle: ", end="")
target_75 = cumulative_resolved[-1] * 0.75
for i, val in enumerate(cumulative_resolved):
    if val >= target_75:
        print(f"{i+1}")
        break

plt.show()
