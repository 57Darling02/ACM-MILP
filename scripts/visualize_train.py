import argparse
import os

from vis_utils import (
    ensure_dir,
    load_scalar_history_from_events,
    parse_loss_history_from_log,
    plot_training_loss,
    write_manifest,
)


def resolve_log_path(train_dir: str, explicit_log_path: str = None) -> str:
    if explicit_log_path:
        return explicit_log_path

    candidates = [
        os.path.join(train_dir, "train.log"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    for file_name in os.listdir(train_dir):
        if file_name.endswith(".log"):
            return os.path.join(train_dir, file_name)
    raise FileNotFoundError("No log file found under `{0}`.".format(train_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training loss from TensorBoard events or fallback logs."
    )
    parser.add_argument("--train-dir", required=True, help="Training run directory under outputs/train")
    parser.add_argument("--tag", default="Train/total_loss", help="TensorBoard scalar tag to plot")
    parser.add_argument("--event-path", default=None, help="Optional event file or directory override")
    parser.add_argument("--log-path", default=None, help="Optional training log override")
    parser.add_argument("--output-dir", default=None, help="Where to save plots")
    parser.add_argument("--smooth-window", type=int, default=50, help="Rolling window for smoothed loss curve")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.train_dir, "visualizations")
    ensure_dir(output_dir)

    history = None
    source = None
    event_path = args.event_path or args.train_dir
    try:
        history = load_scalar_history_from_events(event_path, args.tag)
        source = "tensorboard"
    except Exception as event_error:
        log_path = resolve_log_path(args.train_dir, args.log_path)
        history = parse_loss_history_from_log(log_path)
        source = "log"
        print("TensorBoard parsing skipped:", event_error)
        print("Falling back to training log:", log_path)

    history_csv = os.path.join(output_dir, "loss_curve.csv")
    history.to_csv(history_csv, index=False)

    figure_path = os.path.join(output_dir, "training_loss.png")
    plot_training_loss(
        history=history,
        output_path=figure_path,
        title="Training Loss",
        smooth_window=args.smooth_window,
        dpi=args.dpi,
    )

    write_manifest(
        os.path.join(output_dir, "manifest.json"),
        {
            "stage": "train",
            "train_dir": args.train_dir,
            "tag": args.tag,
            "source": source,
            "rows": int(len(history)),
            "generated_files": [history_csv, figure_path],
        },
    )

    print("Saved training visualizations to:", output_dir)


if __name__ == "__main__":
    main()
