"""
CSL 7640: Natural Language Understanding — Assignment 2
Problem 2, TASK-2: Quantitative Evaluation

Metrics:
  - Novelty Rate : % of generated names NOT in the training set
  - Diversity    : unique generated names / total generated names

Usage:
  python evaluate_models.py

"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# FILE PATHS


TRAINING_FILE = "TrainingNames.txt"

GENERATED_FILES = {
    "Vanilla RNN":        "generated_rnn.txt",
    "Bidirectional LSTM": "generated_blstm.txt",
    "RNN + Attention":    "generated_attention.txt",
}



# HELPERS


def load_names(filepath: str) -> list[str]:
    """Load names from a text file, one per line. Strips whitespace."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")
    with open(filepath, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def novelty_rate(generated: list[str], training: list[str]) -> float:
    """
    Novelty Rate = (names NOT in training set / total generated) * 100
    Comparison is case-insensitive.
    """
    training_set = set(n.lower() for n in training)
    novel_count  = sum(1 for n in generated if n.lower() not in training_set)
    return novel_count / len(generated) * 100 if generated else 0.0


def diversity(generated: list[str]) -> float:
    """
    Diversity = (unique generated names / total generated names) * 100
    Comparison is case-insensitive.
    """
    unique = set(n.lower() for n in generated)
    return len(unique) / len(generated) * 100 if generated else 0.0


def compute_metrics(generated: list[str], training: list[str]) -> dict:
    """Return all metrics for one model as a dict."""
    return {
        "total_generated": len(generated),
        "unique_names":    len(set(n.lower() for n in generated)),
        "novelty_rate":    round(novelty_rate(generated, training), 2),
        "diversity":       round(diversity(generated), 2),
    }



# PRINT TABLE


def print_results_table(results: dict):
    """Print a clean comparison table to the console."""
    col_w = 22   # column width

    header = (f"{'Model':<{col_w}} {'Total':>8} {'Unique':>8} "
              f"{'Novelty (%)':>13} {'Diversity (%)':>14}")
    sep    = "─" * len(header)

    print("\n" + sep)
    print("  TASK-2: QUANTITATIVE EVALUATION RESULTS")
    print(sep)
    print(header)
    print(sep)

    for model_name, m in results.items():
        print(f"{model_name:<{col_w}} "
              f"{m['total_generated']:>8} "
              f"{m['unique_names']:>8} "
              f"{m['novelty_rate']:>13.2f} "
              f"{m['diversity']:>14.2f}")

    print(sep)

    #  Best model per metric 
    best_novelty   = max(results, key=lambda k: results[k]["novelty_rate"])
    best_diversity = max(results, key=lambda k: results[k]["diversity"])
    print(f"\n  Best Novelty Rate  → {best_novelty}  "
          f"({results[best_novelty]['novelty_rate']}%)")
    print(f"  Best Diversity     → {best_diversity}  "
          f"({results[best_diversity]['diversity']}%)")
    print(sep + "\n")



# BAR CHART


def plot_comparison(results: dict, save_path: str = "task2_evaluation.png"):
    """
    Side-by-side grouped bar chart comparing Novelty Rate and Diversity
    across all three models.
    """
    model_names  = list(results.keys())
    novelty_vals = [results[m]["novelty_rate"]  for m in model_names]
    diversity_vals = [results[m]["diversity"]   for m in model_names]

    x      = np.arange(len(model_names))
    width  = 0.35
    colours = ["#e74c3c", "#3498db"]   # red for novelty, blue for diversity

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - width/2, novelty_vals,   width, label="Novelty Rate (%)",
                   color=colours[0], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, diversity_vals, width, label="Diversity (%)",
                   color=colours[1], alpha=0.85, edgecolor="white")

    # Annotate bar values on top
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=colours[0])
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=colours[1])

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Task-2: Novelty Rate & Diversity Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved bar chart → '{save_path}'")


# PRINT SAMPLE NAMES PER MODEL


def print_samples(generated_dict: dict, n: int = 15):
    """Print a few generated names from each model."""
    print("─" * 60)
    print("  SAMPLE GENERATED NAMES")
    print("─" * 60)
    for model_name, names in generated_dict.items():
        print(f"\n  [{model_name}]")
        print("  " + ",  ".join(names[:n]))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load training names
    training_names = load_names(TRAINING_FILE)
    print(f"[Data] Training set: {len(training_names)} names")

    # Load generated names for each model
    generated_dict = {}
    for model_name, filepath in GENERATED_FILES.items():
        gen = load_names(filepath)
        generated_dict[model_name] = gen
        print(f"[Data] {model_name}: {len(gen)} generated names loaded")

    # Compute metrics
    results = {}
    for model_name, gen in generated_dict.items():
        results[model_name] = compute_metrics(gen, training_names)

    # Print table
    print_results_table(results)

    # Print samples
    print_samples(generated_dict)

    # Plot bar chart
    plot_comparison(results)


if __name__ == "__main__":
    main()