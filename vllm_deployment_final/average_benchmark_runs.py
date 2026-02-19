"""
Average multiple benchmark JSON files for scientific reporting.

For each (model, framework, concurrency) configuration, computes:
  - mean and std of every metric across runs
  - number of runs averaged (n)

Usage:
    python average_benchmark_runs.py
"""

import json
import os
import numpy as np
from collections import defaultdict

# Input files (5 runs)
INPUT_FILES = [
    "vllm_rag_benchmark_20260215_055742.json",
    "vllm_rag_benchmark_20260215_063425.json",
    "vllm_rag_benchmark_20260215_071130.json",
    "vllm_rag_benchmark_20260215_075248.json",
    "vllm_rag_benchmark_20260215_082805.json",
]

OUTPUT_FILE = "vllm_rag_benchmark_averaged_5runs.json"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_runs(files):
    """Load all benchmark JSON files."""
    all_runs = []
    for f in files:
        path = os.path.join(SCRIPT_DIR, f)
        with open(path) as fh:
            all_runs.append(json.load(fh))
        print(f"  Loaded {f} ({len(all_runs[-1])} entries)")
    return all_runs


def average_runs(all_runs):
    """Average metrics across runs for each (model, framework, concurrency) key."""
    # Group by (model, framework, concurrency)
    groups = defaultdict(list)
    for run in all_runs:
        for entry in run:
            key = (entry["model"], entry["framework"], entry["concurrency"])
            groups[key].append(entry["stats"])

    results = []
    for (model, framework, concurrency), stats_list in sorted(groups.items()):
        n = len(stats_list)

        # Collect all stat keys (skip None values)
        all_keys = set()
        for s in stats_list:
            all_keys.update(s.keys())

        avg_stats = {}
        std_stats = {}

        for key in sorted(all_keys):
            values = []
            for s in stats_list:
                v = s.get(key)
                if v is not None and isinstance(v, (int, float)):
                    values.append(float(v))

            if values:
                avg_stats[key] = float(np.mean(values))
                std_stats[key] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            else:
                avg_stats[key] = None
                std_stats[key] = None

        results.append({
            "model": model,
            "framework": framework,
            "concurrency": concurrency,
            "n_runs": n,
            "stats_mean": avg_stats,
            "stats_std": std_stats,
        })

    return results


def main():
    print(f"Averaging {len(INPUT_FILES)} benchmark runs for scientific reporting\n")

    all_runs = load_runs(INPUT_FILES)

    # Verify all runs have the same configurations
    configs_per_run = [
        set((e["model"], e["framework"], e["concurrency"]) for e in run)
        for run in all_runs
    ]
    common = configs_per_run[0]
    for i, c in enumerate(configs_per_run[1:], 2):
        if c != common:
            missing = common - c
            extra = c - common
            if missing:
                print(f"  Warning: Run {i} missing configs: {missing}")
            if extra:
                print(f"  Warning: Run {i} has extra configs: {extra}")
        common = common & c

    print(f"\n  Common configurations across all runs: {len(common)}")
    print(f"  Models: {sorted(set(m for m, _, _ in common))}")
    print(f"  Frameworks: {sorted(set(f for _, f, _ in common))}")
    print(f"  Concurrency levels: {sorted(set(c for _, _, c in common))}")

    results = average_runs(all_runs)

    # Save
    output_path = os.path.join(SCRIPT_DIR, OUTPUT_FILE)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved averaged results to {OUTPUT_FILE}")
    print(f"  {len(results)} configurations, each averaged over up to {len(INPUT_FILES)} runs")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'Model':<20} {'Framework':<12} {'Conc':>5} {'N':>3} {'Avg Lat (s)':>12} {'± Std':>8} {'RPS':>8} {'± Std':>8} {'Tok/s':>8} {'± Std':>8}")
    print(f"{'='*100}")
    for r in results:
        m = r["stats_mean"]
        s = r["stats_std"]
        print(f"{r['model']:<20} {r['framework']:<12} {r['concurrency']:>5} {r['n_runs']:>3} "
              f"{m['avg_latency']:>12.2f} {s['avg_latency']:>8.2f} "
              f"{m['requests_per_second']:>8.2f} {s['requests_per_second']:>8.2f} "
              f"{m['tokens_per_second']:>8.1f} {s['tokens_per_second']:>8.1f}")


if __name__ == "__main__":
    main()
