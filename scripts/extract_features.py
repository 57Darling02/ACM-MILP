import argparse
import json
import os
import sys
import numpy as np
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.benchmarks.utils import compute_features
from src.utils import instance2graph
from tqdm import tqdm

def to_builtin(obj):
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    if args.serial:
        files = os.listdir(args.instances_dir)
        features = []
        for f in tqdm(files, desc="Computing features (serial)"):
            p = os.path.join(args.instances_dir, f)
            try:
                _, feat = instance2graph(p, compute_features=True, comm_detec=False)
                features.append(feat)
            except Exception as e:
                print(f"[WARN] Skip {p}: {e}")
    else:
        features = compute_features(args.instances_dir, num_workers=args.num_workers)
    features = to_builtin(features)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()


# use script by command :
# python ./scripts/extract_features.py --instances-dir ./data/mis/test --output-json ./output/features.json
