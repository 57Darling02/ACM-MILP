import argparse
import os

from vis_utils import (
    compute_js_divergence_from_series,
    ensure_dir,
    load_csv,
    pick_feature_columns,
    plot_feature_suite,
    plot_hardness_suite,
    write_manifest,
)


def format_js_annotations(dataframe_map, columns):
    if "generated" not in dataframe_map or "reference" not in dataframe_map:
        return {}, {}

    js_values = {}
    js_annotations = {}
    generated_df = dataframe_map["generated"]
    reference_df = dataframe_map["reference"]

    for column in columns:
        if column not in generated_df.columns or column not in reference_df.columns:
            continue
        js_value = compute_js_divergence_from_series(generated_df[column], reference_df[column])
        if js_value is None:
            continue
        js_values[column] = js_value
        js_annotations[column] = "JSD={0:.4f}".format(js_value)

    return js_values, js_annotations


def main():
    parser = argparse.ArgumentParser(
        description="Visualize generated-instance benchmark outputs with KDE, hardness plots, and JS divergence against preprocess statistics."
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

    feature_js_values, feature_js_annotations = format_js_annotations(dataframe_map, columns)
    hardness_js_values, hardness_js_annotations = format_js_annotations(hardness_map, ["solving_time", "num_nodes"])

    feature_paths = plot_feature_suite(
        dataframe_map=dataframe_map,
        output_dir=feature_output_dir,
        columns=columns,
        title_prefix="Generate",
        bins=args.bins,
        dpi=args.dpi,
        annotations=feature_js_annotations,
    )
    hardness_paths = plot_hardness_suite(
        dataframe_map=hardness_map,
        output_dir=hardness_output_dir,
        title_prefix="Generate",
        bins=args.bins,
        dpi=args.dpi,
        annotations=hardness_js_annotations,
    )

    js_summary = {
        "features": feature_js_values,
        "hardness": hardness_js_values,
        "feature_mean_jsd": (sum(feature_js_values.values()) / len(feature_js_values)) if feature_js_values else None,
        "hardness_mean_jsd": (sum(hardness_js_values.values()) / len(hardness_js_values)) if hardness_js_values else None,
    }
    js_summary_path = None
    if feature_js_values or hardness_js_values:
        js_summary_path = os.path.join(output_dir, "js_divergence.json")
        write_manifest(js_summary_path, js_summary)

    manifest_payload = {
        "stage": "generate",
        "benchmark_dir": args.benchmark_dir,
        "features_csv": features_csv,
        "solving_results_csv": solving_csv,
        "reference_features_csv": reference_features_csv,
        "reference_solving_results_csv": reference_solving_csv,
        "feature_columns": columns,
        "js_divergence": js_summary,
        "generated_files": feature_paths + hardness_paths,
    }
    if js_summary_path is not None:
        manifest_payload["generated_files"].append(js_summary_path)

    write_manifest(
        os.path.join(output_dir, "manifest.json"),
        manifest_payload,
    )

    print("Saved generate visualizations to:", output_dir)


if __name__ == "__main__":
    main()
