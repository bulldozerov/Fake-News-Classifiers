import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from process_data import RAW_FILE, TOP_N, bootstrap_ci, get_values, load_benchmark

FIG_DIR = "reports/figures"
COLORS = sns.color_palette("colorblind")

matplotlib.use("Agg")

def save_figure(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_1(shares):
    fig, ax = plt.subplots(figsize=(6, 4))
    styles = [("raw", "Original text", "o", "-", COLORS[0]),
              ("clean", "Cleaned text", "s", "--", COLORS[1])]
    
    for version, label, marker, linestyle, color in styles:
        part = shares[shares["text_version"] == version].sort_values("seed")
        means, lows, highs = [], [], []
        for n in TOP_N:
            values = part[f"top{n}_source_share"].values * 100  # to percent
            low, high = bootstrap_ci(values)
            means.append(values.mean())
            lows.append(low)
            highs.append(high)
        means, lows, highs = np.array(means), np.array(lows), np.array(highs)

        # 95% CI: error bars + shaded band
        ax.errorbar(TOP_N, means, yerr=[means - lows, highs - means], color=color,
                    marker=marker, linestyle=linestyle, capsize=4, label=label)
        ax.fill_between(TOP_N, lows, highs, color=color, alpha=0.2)

        # the share in the whole vocabulary (what we expect without any bias)
        vocab = part["vocab_source_share"].mean() * 100
        ax.axhline(vocab, color=color, linestyle=":", linewidth=1,
                   label=f"{label}: whole vocabulary")
        
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(TOP_N)
    ax.set_xticklabels([str(n) for n in TOP_N])
    ax.set_xlabel("Top-N features by absolute weight (number of features)")
    ax.set_ylabel("Source-specific features (% of top-N)")
    ax.set_title("Share of source-specific features (mean, 95% CI, 10 seeds)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    save_figure(fig, "fig1_source_shares")  # file name is the one required by the task


def draw_box(ax, values, x, color, hatch):
    box = ax.boxplot(values, positions=[x], widths=0.6, patch_artist=True,
                     showfliers=False, medianprops={"color": "black"})
    box["boxes"][0].set_facecolor(color)
    box["boxes"][0].set_alpha(0.6)
    box["boxes"][0].set_hatch(hatch)

    jitter = np.linspace(-0.15, 0.15, len(values))
    ax.scatter(x + jitter, values, color="black", s=10, zorder=3)

    low, high = bootstrap_ci(values)
    mean = values.mean()
    ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], color="black", marker="D",
                markersize=5, capsize=4, zorder=4)


def figure_2(metrics, majority_f1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    for ax, test_set, title in [(ax1, "isot_test", "ISOT test set"), (ax2, "liar", "LIAR")]:
        draw_box(ax, get_values(metrics, "raw", test_set, "f1"), 1, COLORS[0], "")
        draw_box(ax, get_values(metrics, "clean", test_set, "f1"), 2, COLORS[1], "//")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Original\ntext", "Cleaned\ntext"])
        ax.set_xlabel("Text version")
        ax.set_ylabel("Macro-F1 (score from 0 to 1)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    ax2.axhline(majority_f1, color=COLORS[3], linestyle="-.")
    ax2.set_ylim(majority_f1 - 0.02, 0.47)
    ax2.legend(handles=[Line2D([], [], color=COLORS[3], linestyle="-.",
                               label="Majority-class baseline")],
               loc="upper right", fontsize=8)

    fig.suptitle("Macro-F1 over 10 seeds (black diamond = mean with 95% CI, dots = seeds)")
    fig.tight_layout()
    save_figure(fig, "fig2_distribution_boxplot")


def main():
    np.random.seed(42)  # same seed as in process_data.py
    
    df = load_benchmark(RAW_FILE)
    metrics = df[(df["record_type"] == "metrics") & (df["model"] == "tfidf_logreg")]
    shares = df[df["record_type"] == "shares"]
    majority_f1 = df[df["model"] == "majority_class"]["f1"].iloc[0]

    figure_1(shares)
    figure_2(metrics, majority_f1)
    print(f"Figures saved to {FIG_DIR}/")
    
if __name__ == "__main__":
    main()