import argparse
import os

from vis_utils import (
    ensure_dir,
    load_csv,
    pick_feature_columns,
    plot_feature_suite,
    plot_hardness_suite,
    write_manifest,
)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize generated-instance benchmark outputs with KDE and hardness plots."
    )
    parser.add_argument("--benchmark-dir", required=True, help="Directory like outputs/train/.../eta-0.1/benchmark_step_500")
    parser.add_argument("--features-csv", default=None, help="Optional override for benchmark features.csv")
    parser.add_argument("--solving-results-csv", default=None, help="Optional override for benchmark solving_results.csv")
    parser.add_argument("--reference-stats-dir", default=None, help="Optional preprocess stats directory for overlay comparison")
    parser.add_argument("--reference-features-csv", default=None, help="Optional override for reference features.csv")
    parser.add_argument("--reference-solving-results-csv", default=None, help="Optional override for reference solving_results.csv")
    parser.add_argument("--output-dir", default=None, help="Where to save plots")
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional subset of feature columns to visualize. Default uses common graph statistics columns.",
    )
    args = parser.parse_args()

    features_csv = args.features_csv or os.path.join(args.benchmark_dir, "features.csv")
    solving_csv = args.solving_results_csv or os.path.join(args.benchmark_dir, "solving_results.csv")
    output_dir = args.output_dir or os.path.join(args.benchmark_dir, "visualizations")
    feature_output_dir = ensure_dir(os.path.join(output_dir, "features"))
    hardness_output_dir = ensure_dir(os.path.join(output_dir, "hardness"))

    dataframe_map = {"generated": load_csv(features_csv)}
    hardness_map = {"generated": load_csv(solving_csv)}

    if args.reference_stats_dir or args.reference_features_csv or args.reference_solving_results_csv:
        reference_features_csv = args.reference_features_csv or os.path.join(args.reference_stats_dir, "features.csv")
        reference_solving_csv = args.reference_solving_results_csv or os.path.join(args.reference_stats_dir, "solving_results.csv")
        dataframe_map["reference"] = load_csv(reference_features_csv)
        hardness_map["reference"] = load_csv(reference_solving_csv)
    else:
        reference_features_csv = None
        reference_solving_csv = None

    generated_columns = pick_feature_columns(dataframe_map["generated"], args.columns)
    common_columns = [
        column for column in generated_columns
        if all(column in df.columns for df in dataframe_map.values())
    ]
    columns = common_columns if len(dataframe_map) > 1 else generated_columns

    feature_paths = plot_feature_suite(
        dataframe_map=dataframe_map,
        output_dir=feature_output_dir,
        columns=columns,
        title_prefix="Generate",
        bins=args.bins,
        dpi=args.dpi,
    )
    hardness_paths = plot_hardness_suite(
        dataframe_map=hardness_map,
        output_dir=hardness_output_dir,
        title_prefix="Generate",
        bins=args.bins,
        dpi=args.dpi,
    )

    write_manifest(
        os.path.join(output_dir, "manifest.json"),
        {
            "stage": "generate",
            "benchmark_dir": args.benchmark_dir,
            "features_csv": features_csv,
            "solving_results_csv": solving_csv,
            "reference_features_csv": reference_features_csv,
            "reference_solving_results_csv": reference_solving_csv,
            "feature_columns": columns,
            "generated_files": feature_paths + hardness_paths,
        },
    )

    print("Saved generate visualizations to:", output_dir)


if __name__ == "__main__":
    main()
