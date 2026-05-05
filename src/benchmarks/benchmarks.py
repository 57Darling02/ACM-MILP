import json
import logging
import os

import pandas as pd
from omegaconf import DictConfig, OmegaConf


from .utils import FEATURES, compute_features, compute_jsdiv, solve_instances


class Benchmark():
    def __init__(self, config: DictConfig, dataset_stats_dir: str):
        self.config = config
        self.dataset_stats_dir = dataset_stats_dir

    def assess_samples(self,
                       samples_dir: str,
                       benchmark_dir: str
                       ):
        """
        Assess the generated samples.

        Args:
            samples_dir: directory of generated samples
            benchmark_dir: directory to save benchmark results

        Returns:
            results: dict, benchmark results
        """
        os.makedirs(benchmark_dir, exist_ok=True)
        distribution_results = None
        if not bool(getattr(self.config, "skip_distribution", False)):
            distribution_results = assess_distribution(
                self.config, samples_dir, self.dataset_stats_dir, benchmark_dir)

        solving_results = None
        if not bool(getattr(self.config, "skip_solving", False)):
            solving_results = assess_solving_results(
                self.config, samples_dir, self.dataset_stats_dir, benchmark_dir)
        results = {
            "distribution": distribution_results,
            "solving": solving_results,
        }
        return results

    @staticmethod
    def log_info(
        generator_config: DictConfig,
        benchmarking_config: DictConfig,
        meta_results: dict,
        save_path: str
    ):
        info = {
            "generator": OmegaConf.to_container(generator_config),
            "benchmarking": OmegaConf.to_container(benchmarking_config),
            "meta_results": meta_results,
        }
        logging.info("-"*10 + " Benchmarking  Info " + "-"*10)
        json_msg = json.dumps(info, indent=4)
        for msg in json_msg.split("\n"):
            logging.info(msg)
        logging.info("-"*40)

        if save_path is not None:
            with open(save_path, 'w') as f:
                f.write(json.dumps(info, indent=4))
            logging.info(f"Benchmarking info saved to {save_path} .")


def assess_distribution(config, samples_dir, dataset_stats_dir, benchmark_dir):
    features_path = os.path.join(benchmark_dir, "features.csv")
    reuse_existing = bool(getattr(config, "reuse_existing", False))

    if reuse_existing and os.path.exists(features_path):
        logging.info("Reusing existing feature cache at %s", features_path)
        samples_features = pd.read_csv(features_path).set_index("instance")
    else:
        samples_features = compute_features(
            samples_dir,
            num_workers=config.num_workers,
            max_instances=getattr(config, "max_instances", None),
        )
        samples_features = pd.DataFrame(samples_features).set_index("instance")
        samples_features.to_csv(features_path)

    reference_features = pd.read_csv(os.path.join(
        dataset_stats_dir, "features.csv")).set_index("instance")

    used_features = list(FEATURES.keys())
    samples_features = samples_features.loc[:, used_features].to_numpy()
    reference_features = reference_features.loc[:, used_features].to_numpy()

    score, meta_results = compute_jsdiv(
        samples_features, reference_features, num_samples=config.num_samples)

    results = {
        "score": score,
        "meta_results": meta_results,
    }
    return results


def assess_solving_results(config, samples_dir, dataset_stats_dir, benchmark_dir):
    solving_results_path = os.path.join(benchmark_dir, "solving_results.csv")
    reuse_existing = bool(getattr(config, "reuse_existing", False))

    if reuse_existing and os.path.exists(solving_results_path):
        logging.info("Reusing existing solving cache at %s", solving_results_path)
        samples_solving_results = pd.read_csv(solving_results_path).set_index("instance")
    else:
        samples_solving_results = solve_instances(
            samples_dir,
            num_workers=config.num_workers,
            mip_gap=config.solver.mip_gap,
            time_limit=config.solver.time_limit,
            max_instances=getattr(config, "max_instances", None),
        )
        samples_solving_results = pd.DataFrame(
            samples_solving_results).set_index("instance")
        samples_solving_results.to_csv(solving_results_path)

    reference_solving_results_path = os.path.join(
        dataset_stats_dir, "solving_results.csv")
    if not os.path.exists(reference_solving_results_path):
        logging.warning(
            "Skip solving benchmark because reference file is missing: %s",
            reference_solving_results_path,
        )
        return {
            "skipped": True,
            "reason": f"missing reference file: {reference_solving_results_path}",
        }

    reference_solving_results = pd.read_csv(
        reference_solving_results_path).set_index("instance")

    samples_solving_time = samples_solving_results.loc[:, [
        "solving_time"]].to_numpy()
    samples_num_nodes = samples_solving_results.loc[:, [
        "num_nodes"]].to_numpy()
    reference_solving_time = reference_solving_results.loc[:, [
        "solving_time"]].to_numpy()
    reference_num_nodes = reference_solving_results.loc[:, [
        "num_nodes"]].to_numpy()

    results = {
        "solving_time": {
            "mean": samples_solving_time.mean(),
            "mean_reference": reference_solving_time.mean(),
            "mean_error": abs(samples_solving_time.mean() - reference_solving_time.mean()) / reference_solving_time.mean(),
        },
        "num_nodes": {
            "mean": samples_num_nodes.mean(),
            "mean_reference": reference_num_nodes.mean(),
            "mean_error": abs(samples_num_nodes.mean() - reference_num_nodes.mean()) / reference_num_nodes.mean(),
        },
    }
    return results
