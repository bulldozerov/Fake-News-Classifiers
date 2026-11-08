import os

import numpy as np
import pandas as pd
from scipy import stats

RAW_FILE = "data/raw/test_run.jsonl"
OUT_DIR = "data/processed"
N_BOOT = 10000
MAX_DROP = 0.02
TOP_N = [100, 500, 1000]


def load_benchmark(path):
    df = pd.read_json(path, lines=True)
    df = df[df["phase"] == "benchmark"] #without warm-up
    return df


def bootstrap_ci(values):
    # 95% confidence interval of mean
    means = []
    for _ in range(N_BOOT):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def describe(values):
    # mean, SD, median, percentiles and CI
    low, high = bootstrap_ci(values)
    return {
        "n": len(values),
        "mean": np.mean(values),
        "sd": np.std(values, ddof=1),
        "median": np.median(values),
        "p25": np.percentile(values, 25),
        "p75": np.percentile(values, 75),
        "p95": np.percentile(values, 95),
        "p99": np.percentile(values, 99),
        "ci95_low": low,
        "ci95_high": high,
    }


def get_values(metrics, version, test_set, metric):
    # values of one metric for one condition, one number per seed
    part = metrics[(metrics["text_version"] == version) & (metrics["test_set"] == test_set)]
    part = part.sort_values("seed")
    return part[metric].values


def main():
    np.random.seed(42)  # fixed seed

    df = load_benchmark(RAW_FILE)
    metrics = df[(df["record_type"] == "metrics") & (df["model"] == "tfidf_logreg")]
    majority = df[df["model"] == "majority_class"]
    shares = df[df["record_type"] == "shares"]

    # ---------- 1. summary table (mean, SD, median, p95, p99, CI)
    summary_rows = []
    for version in ["raw", "clean"]:
        for test_set in ["isot_test", "liar"]:
            for metric in ["f1", "accuracy"]:
                values = get_values(metrics, version, test_set, metric)
                row = {"text_version": version, "test_set": test_set, "metric": metric}
                row.update(describe(values))
                summary_rows.append(row)
    for metric in ["f1", "accuracy"]:  # majority baseline: only one number, no seeds
        summary_rows.append({"text_version": "majority_class", "test_set": "liar",
                             "metric": metric, "n": 1, "mean": majority[metric].iloc[0]})
    summary = pd.DataFrame(summary_rows)

    # ---------- 2. hypothesis tests
    tests = []

            # H1: ISOT test is better than LIAR (raw text)
    for metric in ["f1", "accuracy"]:
        isot = get_values(metrics, "raw", "isot_test", metric)
        liar = get_values(metrics, "raw", "liar", metric)
        drop = isot - liar  # one number per seed
        low, high = bootstrap_ci(drop)
        p = stats.ttest_rel(isot, liar, alternative="greater").pvalue
        sd = max(np.std(isot, ddof=1), np.std(liar, ddof=1))
        tests.append({"hypothesis": "H1", "what": f"{metric}: ISOT minus LIAR (raw)",
                      "estimate": drop.mean(), "ci95_low": low, "ci95_high": high,
                      "p_value": p, "supported": bool(p < 0.05 and drop.mean() > sd)})

            # H2: share of source-specific features in top-N is bigger than in the whole vocabulary
    for version in ["raw", "clean"]:
        part = shares[shares["text_version"] == version].sort_values("seed")
        vocab_shares = part["vocab_source_share"].values
        for n in TOP_N:
            top_shares = part[f"top{n}_source_share"].values
            p_values = []
            for top_share, vocab_share in zip(top_shares, vocab_shares):
                k = round(top_share * n)  # how many of the top-N are source-specific
                p_values.append(stats.binomtest(k, n, vocab_share, alternative="greater").pvalue)
            worst_p = max(p_values)  # the weakest seed
            tests.append({"hypothesis": "H2", "what": f"{version}: share in top-{n}",
                          "estimate": top_shares.mean(), "p_value": worst_p,
                          "supported": bool(worst_p < 0.05),
                          "note": f"vocabulary share = {vocab_shares.mean():.4f}"})

            # H3: cleaning increases F1 on LIAR
    diff = get_values(metrics, "clean", "liar", "f1") - get_values(metrics, "raw", "liar", "f1")
    low, high = bootstrap_ci(diff)
    sd = np.std(get_values(metrics, "raw", "liar", "f1"), ddof=1)
    tests.append({"hypothesis": "H3", "what": "F1 on LIAR: clean minus raw",
                  "estimate": diff.mean(), "ci95_low": low, "ci95_high": high,
                  "supported": bool(low > 0 and diff.mean() > sd),
                  "note": f"seeds with gain > 0: {(diff > 0).sum()} of {len(diff)}"})

    # guardrails
    isot_drop = get_values(metrics, "raw", "isot_test", "f1") - get_values(metrics, "clean", "isot_test", "f1")
    low, high = bootstrap_ci(isot_drop)
    tests.append({"hypothesis": "Guardrail", "what": "F1 on ISOT test: raw minus clean",
                  "estimate": isot_drop.mean(), "ci95_low": low, "ci95_high": high,
                  "supported": bool(isot_drop.max() <= MAX_DROP),
                  "note": f"max drop = {isot_drop.max():.4f}, limit = {MAX_DROP}"})
    overlap = metrics.drop_duplicates(subset="seed")["train_test_overlap"].sum()
    tests.append({"hypothesis": "Guardrail", "what": "texts in both ISOT train and test",
                  "estimate": overlap, "supported": bool(overlap == 0)})
    tests = pd.DataFrame(tests)

    # save and print
    os.makedirs(OUT_DIR, exist_ok=True)
    summary.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)
    tests.to_csv(os.path.join(OUT_DIR, "hypothesis_tests.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(summary.round(4).to_string(index=False))
    print()
    print(tests.drop(columns="note").round(4).to_string(index=False))


if __name__ == "__main__":
    main()