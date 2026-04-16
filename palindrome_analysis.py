import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ALIAS = {
    "RelaxedKaczmarzQNorm": "KDA",
    "GatedDeltaNet": "GDN",
    "Mamba2": "Mamba2",
}

MODEL_STYLE = {
    "KDA": {"color": "#5167ff", "linestyle": "-", "marker": "o"},
    "GDN": {"color": "#0b8f8b", "linestyle": "--", "marker": "*"},
    "Mamba2": {"color": "#ff8a3d", "linestyle": "--", "marker": "x"},
}

MODEL_ORDER = ["KDA", "GDN", "Mamba2"]


def normalize_model_name(model_name: str) -> str:
    for prefix, alias in MODEL_ALIAS.items():
        if model_name.startswith(prefix):
            return alias
    return model_name


def load_results(out_root: str):
    result_files = glob.glob(os.path.join(out_root, "**", "results.json"), recursive=True)

    seq_rows = []
    step_rows = []

    for path in result_files:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skip invalid result file {path}: {e}")
            continue

        model_raw = data.get("model_name", "Unknown")
        model = normalize_model_name(model_raw)
        args = data.get("args", {})
        seed = args.get("seed")

        eval_acc_by_seq_len = data.get("eval_acc_by_seq_len", {})
        for seq_len_str, acc in eval_acc_by_seq_len.items():
            seq_rows.append(
                {
                    "Model": model,
                    "Model Raw": model_raw,
                    "Seq Len": int(seq_len_str),
                    "Accuracy": float(acc) * 100.0,
                    "Seed": seed,
                    "Path": path,
                }
            )

        val_history = data.get("val_history", [])
        for item in val_history:
            if "step" not in item or "acc" not in item:
                continue
            step_rows.append(
                {
                    "Model": model,
                    "Model Raw": model_raw,
                    "Step": int(item["step"]),
                    "Accuracy": float(item["acc"]) * 100.0,
                    "Seed": seed,
                    "Path": path,
                }
            )

    return pd.DataFrame(seq_rows), pd.DataFrame(step_rows), result_files


def aggregate_for_plot(df: pd.DataFrame, group_cols):
    if df.empty:
        return df
    return df.groupby(group_cols, as_index=False)["Accuracy"].mean()


def plot_seq_len(ax, df: pd.DataFrame):
    if df.empty:
        ax.text(0.5, 0.5, "No sequence-length metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    agg = aggregate_for_plot(df, ["Model", "Seq Len"])

    for model in MODEL_ORDER:
        model_df = agg[agg["Model"] == model].sort_values("Seq Len")
        if model_df.empty:
            continue
        style = MODEL_STYLE.get(model, {})
        ax.plot(
            model_df["Seq Len"],
            model_df["Accuracy"],
            label=model,
            linewidth=1.6,
            markersize=6,
            **style,
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Sequence length")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)


def plot_steps(ax, df: pd.DataFrame):
    if df.empty:
        ax.text(0.5, 0.5, "No training-step metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    agg = aggregate_for_plot(df, ["Model", "Step"])

    for model in MODEL_ORDER:
        model_df = agg[agg["Model"] == model].sort_values("Step")
        if model_df.empty:
            continue
        style = MODEL_STYLE.get(model, {})
        ax.plot(
            model_df["Step"],
            model_df["Accuracy"],
            label=model,
            linewidth=1.6,
            markersize=5,
            **style,
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Training steps")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="./out/palindrome")
    parser.add_argument("--save_dir", type=str, default="./analysis_results/palindrome")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    seq_df, step_df, files = load_results(args.out_root)
    if not files:
        print("No results.json found.")
        return

    seq_plot_df = aggregate_for_plot(seq_df, ["Model", "Seq Len"]) if not seq_df.empty else seq_df
    step_plot_df = aggregate_for_plot(step_df, ["Model", "Step"]) if not step_df.empty else step_df

    if not seq_df.empty:
        seq_df.to_csv(os.path.join(args.save_dir, "seq_len_metrics_raw.csv"), index=False)
        seq_plot_df.to_csv(os.path.join(args.save_dir, "seq_len_metrics_mean.csv"), index=False)
    if not step_df.empty:
        step_df.to_csv(os.path.join(args.save_dir, "step_metrics_raw.csv"), index=False)
        step_plot_df.to_csv(os.path.join(args.save_dir, "step_metrics_mean.csv"), index=False)

    plt.figure(figsize=(7, 9))
    ax1 = plt.subplot(2, 1, 1)
    plot_seq_len(ax1, seq_df)

    ax2 = plt.subplot(2, 1, 2)
    plot_steps(ax2, step_df)

    plt.tight_layout()
    output_path = os.path.join(args.save_dir, "palindrome_metrics.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Loaded {len(files)} result files")
    print(f"Saved plots and tables to {args.save_dir}")


if __name__ == "__main__":
    main()
