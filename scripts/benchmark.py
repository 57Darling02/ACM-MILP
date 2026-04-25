import argparse
import os
import sys

from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import Benchmark, set_cpu_num


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rerun benchmark evaluation for an existing directory of generated MILP samples."
    )
    parser.add_argument("--samples-dir", required=True, help="Directory containing generated .lp/.mps samples.")
    parser.add_argument("--benchmark-dir", required=True, help="Directory where benchmark outputs will be written.")
    parser.add_argument(
        "--dataset-stats-dir",
        default=None,
        help="Reference stats directory containing features.csv and solving_results.csv.",
    )
    parser.add_argument(
        "--dataset",
        default="mis",
        help="Dataset name used to infer --dataset-stats-dir as preprocess/<dataset>/stats when omitted.",
    )
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_stats_dir = args.dataset_stats_dir or os.path.join("preprocess", args.dataset, "stats")

    set_cpu_num(args.num_workers + 1)
    benchmark_config = OmegaConf.create(
        {
            "num_workers": args.num_workers,
            "num_samples": args.num_samples,
        }
    )
    run_info = OmegaConf.create(
        {
            "source_samples_dir": args.samples_dir,
            "dataset_stats_dir": dataset_stats_dir,
        }
    )

    benchmarker = Benchmark(
        config=benchmark_config,
        dataset_stats_dir=dataset_stats_dir,
    )
    results = benchmarker.assess_samples(
        samples_dir=args.samples_dir,
        benchmark_dir=args.benchmark_dir,
    )

    info_path = os.path.join(args.benchmark_dir, "info.json")
    benchmarker.log_info(
        generator_config=run_info,
        benchmarking_config=benchmark_config,
        meta_results=results,
        save_path=info_path,
    )
    print("Saved benchmark outputs to:", args.benchmark_dir)


if __name__ == "__main__":
    main()
