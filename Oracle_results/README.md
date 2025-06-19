# Oracle Analysis (SWE-bench Verified)

This folder contains scripts and outputs for analyzing oracle-level performance across multiple agents evaluated on the SWE-bench Verified benchmark.

---

## Folder Structure

```
Oracle_results/
├── code/
│   ├── cumulative_oracle_analysis.py  # Plot oracle growth as agents are combined
│   └── oracle_merge.py                # Compute oracle statistics and save JSON reports
├── results/
│   ├── oracle_growth_curve.png        # Line plot showing how performance scales with more agents
│   ├── oracle_results.json            # Overall oracle pass rate and top-performing agents
│   └── instance_analysis.json         # Per-agent task distribution and instance overlap
```

---

## Script Descriptions

### `code/cumulative_oracle_analysis.py`

- **Goal**: Analyze how the number of resolved instances increases as you cumulatively add top agents.
- **Input**: Path to the folder of agent results (each with `results.json`).
- **Output**: 
  - `oracle_growth_curve.png`: Line plot of unique resolved instances vs. number of agents added.

**Plot Interpretation**:
- **X-axis**: Number of top agents combined (ranked by individual performance).
- **Y-axis**: Total number of unique instances resolved by the combined set.
- The curve shows diminishing returns as more agents are added.

---

### `code/oracle_merge.py`

- **Goal**: Compute the oracle pass rate (maximum number of unique instances that can be resolved if we always pick the best agent).
- **Outputs**:
  - `oracle_results.json`: 
    - Oracle pass rate
    - Total resolved instances
    - Top 10 agents by individual performance
  - `instance_analysis.json`:
    - Number of instances each agent attempted
    - Sample of all covered instance IDs
    - Participation distribution

---