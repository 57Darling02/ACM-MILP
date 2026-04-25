import argparse
import math
import os

import numpy as np

from vis_utils import (
    compute_js_divergence,
    ensure_dir,
    load_csv,
    pick_feature_columns,
    plot_feature_suite,
    plot_hardness_suite,
    to_numeric_series,
    write_manifest,
)


PAPER_JS_FEATURE_COLUMNS = [
    "coef_dens",
    "var_degree_mean",
    "var_degree_std",
    "cons_degree_mean",
    "cons_degree_std",
    "lhs_mean",
    "lhs_std",
    "rhs_mean",
    "rhs_std",
    "clustering",
    "modularity",
]


def sample_series_for_js(series, num_samples: int, rng: np.random.Generator):
    values = to_numeric_series(series).to_numpy()
    if values.size == 0:
        return values
    indices = rng.choice(np.arange(values.size), size=num_samples, replace=True)
    return values[indices]


def format_paper_js_annotations(dataframe_map, columns, num_samples: int, bins: int, random_seed=None):
    if "generated" not in dataframe_map or "reference" not in dataframe_map:
        return {}, {}, [], {}

    js_values = {}
    similarity_values = {}
    js_annotations = {}
    used_columns = []
    generated_df = dataframe_map["generated"]
    reference_df = dataframe_map["reference"]
    rng = np.random.default_rng(random_seed)

    for column in columns:
        if column not in PAPER_JS_FEATURE_COLUMNS:
            continue
        if column not in generated_df.columns or column not in reference_df.columns:
            continue

        sampled_generated = sample_series_for_js(generated_df[column], num_samples, rng)
        sampled_reference = sample_series_for_js(reference_df[column], num_samples, rng)
        if sampled_generated.size == 0 or sampled_reference.size == 0:
            continue

        js_value = compute_js_divergence(sampled_generated, sampled_reference, bins=bins)
        if js_value is None:
            continue

        similarity = 1.0 - js_value / math.log(2)
        js_values[column] = js_value
        similarity_values[column] = similarity
        js_annotations[column] = "Sim={0:.3f}, JSD={1:.4f}".format(similarity, js_value)
        used_columns.append(column)

    return js_values, similarity_values, used_columns, js_annotations


def main():
    parser = argparse.ArgumentParser(
        description="Visualize generated-instance benchmark outputs with paper-aligned KDE comparison, histogram-based JS divergence, and hardness plots."
    )
    parser.add_argument("--benchmark-dir", required=True, help="Directory like outputs/train/.../eta-0.1/benchmark_step_500")
    parser.add_argument("--features-csv", default=None, help="Optional override for benchmark features.csv")
    parser.add_argument("--solving-results-csv", default=None, help="Optional override for benchmark solving_results.csv")
    parser.add_argument("--reference-stats-dir", default=None, help="Optional preprocess stats directory for overlay comparison")
    parser.add_argument("--reference-features-csv", default=None, help="Optional override for reference features.csv")
    parser.add_argument("--reference-solving-results-csv", default=None, help="Optional override for reference solving_results.csv")
    parser.add_argument("--output-dir", default=None, help="Where to save plots")
    parser.add_argument("--bins", type=int, default=30, help="Histogram bins for visualization only")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--js-num-samples", type=int, default=10000, help="Bootstrap sample count for paper-style JS evaluation")
    parser.add_argument("--js-bins", type=int, default=5, help="Number of shared histogram bins for paper-style JS evaluation")
    parser.add_argument("--js-random-seed", type=int, default=0, help="Random seed for reproducible paper-style JS evaluation")
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

    feature_js_values = {}
    feature_similarity_values = {}
    used_paper_js_columns = []
    feature_js_annotations = {}
    if len(dataframe_map) > 1:
        feature_js_values, feature_similarity_values, used_paper_js_columns, feature_js_annotations = format_paper_js_annotations(
            dataframe_map,
            columns,
            num_samples=args.js_num_samples,
            bins=args.js_bins,
            random_seed=args.js_random_seed,
        )

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
        annotations=None,
    )

    benchmark_summary = {
        "paper_feature_js_divergence": feature_js_values,
        "paper_feature_similarity": feature_similarity_values,
        "paper_feature_similarity_score": (sum(feature_similarity_values.values()) / len(feature_similarity_values)) if feature_similarity_values else None,
        "paper_js_features_used": used_paper_js_columns,
        "js_num_samples": args.js_num_samples,
        "js_bins": args.js_bins,
        "js_random_seed": args.js_random_seed,
    }
    benchmark_summary_path = None
    if feature_js_values or feature_similarity_values:
        benchmark_summary_path = os.path.join(output_dir, "paper_js_benchmark.json")
        write_manifest(benchmark_summary_path, benchmark_summary)

    manifest_payload = {
        "stage": "generate",
        "benchmark_dir": args.benchmark_dir,
        "features_csv": features_csv,
        "solving_results_csv": solving_csv,
        "reference_features_csv": reference_features_csv,
        "reference_solving_results_csv": reference_solving_csv,
        "feature_columns": columns,
        "paper_js_benchmark": benchmark_summary,
        "generated_files": feature_paths + hardness_paths,
    }
    if benchmark_summary_path is not None:
        manifest_payload["generated_files"].append(benchmark_summary_path)

    write_manifest(
        os.path.join(output_dir, "manifest.json"),
        manifest_payload,
    )

    print("Saved generate visualizations to:", output_dir)


if __name__ == "__main__":
    main()

