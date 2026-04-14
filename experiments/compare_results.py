import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "experiments/results"
DATASETS    = ["chexpert", "nih", "combined"]
MODELS      = ["resnet50", "densenet121", "efficientnet_b0", "mobilenet_v3"]
PATHOLOGIES = [
    "pneumonia", "cardiomegaly", "pleural_effusion",
    "pneumothorax", "atelectasis", "lung_mass"
]
OUTPUT_DIR  = "experiments/charts"


def load_results():
    """Load all JSON result files into a nested dict.
    Returns: results[dataset][model] = {metrics, train_curves, val_curves}
    """
    results = {}
    for dataset in DATASETS:
        results[dataset] = {}
        for model in MODELS:
            path = os.path.join(RESULTS_DIR, f"{dataset}_{model}.json")
            if os.path.exists(path):
                with open(path) as f:
                    results[dataset][model] = json.load(f)
            else:
                print(f"  WARNING: Missing {path}")
    return results


def get_metric(results, dataset, model, metric="auc"):
    """Extract per-pathology metric values as a list."""
    records = results[dataset][model]["metrics"]
    return [r[metric] for r in records]


def get_macro(results, dataset, model, metric="auc"):
    """Get macro average of a metric."""
    values = get_metric(results, dataset, model, metric)
    return round(sum(values) / len(values), 3)


def print_model_comparison(results):
    """Table 1: Compare all models per dataset on Macro AUC and F1."""
    print("\n" + "="*60)
    print("TABLE 1: MODEL COMPARISON PER DATASET (Macro AUC)")
    print("="*60)

    for dataset in DATASETS:
        print(f"\n--- {dataset.upper()} ---")
        print(f"{'Model':<20} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
        print("-" * 56)
        for model in MODELS:
            if model in results[dataset]:
                auc = get_macro(results, dataset, model, "auc")
                f1  = get_macro(results, dataset, model, "f1")
                pre = get_macro(results, dataset, model, "precision")
                rec = get_macro(results, dataset, model, "recall")
                print(f"{model:<20} {auc:>8.3f} {f1:>8.3f} {pre:>10.3f} {rec:>8.3f}")


def print_dataset_comparison(results):
    """Table 2: For each model, compare performance across datasets."""
    print("\n" + "="*60)
    print("TABLE 2: DATASET COMPARISON PER MODEL (Macro AUC)")
    print("="*60)

    for model in MODELS:
        print(f"\n--- {model.upper()} ---")
        print(f"{'Dataset':<15} {'AUC':>8} {'F1':>8}")
        print("-" * 33)
        for dataset in DATASETS:
            if model in results.get(dataset, {}):
                auc = get_macro(results, dataset, model, "auc")
                f1  = get_macro(results, dataset, model, "f1")
                print(f"{dataset:<15} {auc:>8.3f} {f1:>8.3f}")


def print_pathology_breakdown(results):
    """Table 3: Per-pathology AUC for best model per dataset."""
    print("\n" + "="*60)
    print("TABLE 3: PER-PATHOLOGY AUC — BEST MODEL PER DATASET")
    print("="*60)

    # TODO: for each dataset, find the model with highest macro AUC
    # then print its per-pathology AUC
    # Hint: use get_macro() to find the best model
    # then use get_metric() to get per-pathology values
    for dataset in DATASETS:
        # Skip if no results for this dataset
        available_models = [m for m in MODELS if m in results[dataset]]
        if not available_models:
            print(f"\n--- {dataset.upper()} --- (no results yet)")
            continue

        # find model with highest macro AUC
        best_model = max(MODELS, 
                         key=lambda m: get_macro(results, dataset, m, "auc")
                         if m in results[dataset] else 0)
        print(f"\n--- {dataset.upper()} - Best: {best_model} ---")
        print(f"{'Pathology':<20} {'AUC':>8}")
        print("-" * 30)
        # get and print per-pathology AUC
        aucs = get_metric(results, dataset, best_model, "auc")
        for pathology, auc in zip(PATHOLOGIES, aucs):
            print(f"{pathology:<20} {auc:>8.3f}")
    pass


def plot_auc_comparison(results):
    """Bar chart: Macro AUC per model, grouped by dataset."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    x = np.arange(len(MODELS))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, dataset in enumerate(DATASETS):
        available_models = [m for m in MODELS if m in results[dataset]]
        if not available_models:
            continue

        aucs = [get_macro(results, dataset, m, "auc")
                if m in results[dataset] else 0
                for m in MODELS]
        bars = ax.bar(x + i * width, aucs, width,
                      label=dataset.capitalize(), color=colors[i], alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Macro AUC")
    ax.set_title("Macro AUC Comparison: Models × Datasets")
    ax.set_xticks(x + width)
    ax.set_xticklabels(MODELS, rotation=15)
    ax.set_ylim(0.5, 0.95)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "auc_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved → {out}")

def plot_pathology_auc(results):
    """Bar chart: Per-pathology AUC for best model on each dataset."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # TODO: for each dataset find the best model
    # get its per-pathology AUC values
    # plot grouped bar chart with pathologies on x-axis
    # one bar per dataset
    # save as "experiments/charts/pathology_auc.png"
    x = np.arange(len(PATHOLOGIES))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    fig, ax = plt.subplots(figsize=(14, 6))

    for i , dataset in enumerate(DATASETS):
        available_models = [m for m in MODELS if m in results[dataset]]
        if not available_models:
            continue

        best_model = max(MODELS,
                         key=lambda m: get_macro(results, dataset, m, "auc")
                         if m in results[dataset] else 0)
        aucs = get_metric(results, dataset, best_model, "auc")
        ax.bar(x + i * width, aucs, width,
               label=f"{dataset} ({best_model})",
               color=colors[i], alpha=0.85)
        
    ax.set_xticks(x + width)
    ax.set_xticklabels(PATHOLOGIES, rotation=20, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("Per-Pathology AUC - Best Model per Dataset")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.70, color="red", linestyle="--", alpha=0.7, label="AUC=0.70 target")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "pathology_auc.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")
    pass


def plot_loss_curves(results):
    """Line chart: Training and validation loss curves."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # One subplot per model, showing all 3 datasets
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = {"chexpert": "#2196F3", "nih": "#4CAF50", "combined": "#FF9800"}

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for dataset in DATASETS:
            if model not in results.get(dataset, {}):
                continue
            # TODO: get train_curves and val_curves from results
            # plot train curve as solid line
            # plot val curve as dashed line
            # use colors[dataset] for color
            # label as f"{dataset} train" and f"{dataset} val"
            train_curves = results[dataset][model]["train_curves"]
            val_curves = results[dataset][model]["val_curves"]
            epochs = range(1, len(train_curves) + 1)
            ax.plot(epochs, train_curves, color=colors[dataset],
                    linestyle="-", label=f"{dataset} train")
            ax.plot(epochs, val_curves, color=colors[dataset],
                    linestyle="--", label=f"{dataset} val", alpha=0.7)
            pass

        ax.set_title(model.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle("Training & Validation Loss Curves", fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "loss_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    print("Loading results...")
    results = load_results()

    print_model_comparison(results)
    print_dataset_comparison(results)
    print_pathology_breakdown(results)

    print("\nGenerating charts...")
    plot_auc_comparison(results)
    plot_pathology_auc(results)
    plot_loss_curves(results)

    print("\nDone.")