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
        description="Visualize preprocess outputs with KDE plots and hardness distributions."
    )
    parser.add_argument("--stats-dir", required=True, help="Directory like preprocess/mis/stats")
    parser.add_argument("--features-csv", default=None, help="Optional override for features.csv")
    parser.add_argument("--solving-results-csv", default=None, help="Optional override for solving_results.csv")
    parser.add_argument("--output-dir", default=None, help="Where to save plots")
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional subset of feature columns to visualize. Default uses available graph statistics columns.",
    )
    args = parser.parse_args()

    features_csv = args.features_csv or os.path.join(args.stats_dir, "features.csv")
    solving_csv = args.solving_results_csv or os.path.join(args.stats_dir, "solving_results.csv")
    output_dir = args.output_dir or os.path.join(args.stats_dir, "visualizations")
    feature_output_dir = ensure_dir(os.path.join(output_dir, "features"))
    hardness_output_dir = ensure_dir(os.path.join(output_dir, "hardness"))

    features_df = load_csv(features_csv)
    solving_df = load_csv(solving_csv)
    columns = pick_feature_columns(features_df, args.columns)

    feature_paths = plot_feature_suite(
        dataframe_map={"preprocess": features_df},
        output_dir=feature_output_dir,
        columns=columns,
        title_prefix="Preprocess",
        bins=args.bins,
        dpi=args.dpi,
    )
    hardness_paths = plot_hardness_suite(
        dataframe_map={"preprocess": solving_df},
        output_dir=hardness_output_dir,
        title_prefix="Preprocess",
        bins=args.bins,
        dpi=args.dpi,
    )

    write_manifest(
        os.path.join(output_dir, "manifest.json"),
        {
            "stage": "preprocess",
            "stats_dir": args.stats_dir,
            "features_csv": features_csv,
            "solving_results_csv": solving_csv,
            "feature_columns": columns,
            "generated_files": feature_paths + hardness_paths,
        },
    )

    print("Saved preprocess visualizations to:", output_dir)


if __name__ == "__main__":
    main()
