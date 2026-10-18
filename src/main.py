import argparse
import logging
import os
import re
import time
import pandas as pd
import yaml

from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def load_isot(true_path, fake_path):
    real = pd.read_csv(true_path)
    fake = pd.read_csv(fake_path)
    real["label"] = 0
    fake["label"] = 1
    data = pd.concat([real, fake], ignore_index=True)
    data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")
    return data[["text", "label"]]


def load_liar(paths, label_map):
    parts = [pd.read_csv(p, sep="\t", header=None, quoting=3, dtype=str) for p in paths]
    data = pd.concat(parts, ignore_index=True)
    data = data.rename(columns={1: "label_6", 2: "text"})
    data["label"] = data["label_6"].map(label_map)
    return data[["text", "label"]]


def clean_text(text, patterns):
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def is_source_specific(feature, source_words):
    return any(w in feature.split() for w in source_words)


def source_specific_share(model, source_words, top_n):
    words = model.named_steps["tfidfvectorizer"].get_feature_names_out()
    weights = model.named_steps["logisticregression"].coef_[0]
    top = sorted(zip(words, weights), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    top_share = sum(is_source_specific(w, source_words) for w, _ in top) / len(top)
    vocab_share = sum(is_source_specific(w, source_words) for w in words) / len(words)
    return top_share, vocab_share


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(cfg["output"]["results_dir"], f"run_{stamp}.log")
    csv_file = os.path.join(cfg["output"]["results_dir"], f"metrics_{stamp}.csv")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


    isot = load_isot(cfg["data"]["isot_true"], cfg["data"]["isot_fake"])
    liar = load_liar(cfg["data"]["liar"], cfg["data"]["liar_label_map"])
    logging.info("ISOT: %d rows, LIAR: %d rows", len(isot), len(liar))

    exp, m, cl = cfg["experiment"], cfg["model"], cfg["cleaning"]
    results = []

    for seed in exp["seeds"]:
        train, test = train_test_split(
            isot, test_size=exp["test_size"], stratify=isot["label"], random_state=seed
        )

        for version in exp["text_versions"]:
            x_train, x_test, x_liar = train["text"], test["text"], liar["text"]
            if version == "clean":
                x_train = x_train.apply(clean_text, patterns=cl["patterns"])
                x_test = x_test.apply(clean_text, patterns=cl["patterns"])
                x_liar = x_liar.apply(clean_text, patterns=cl["patterns"])

            model = make_pipeline(
                TfidfVectorizer(
                    ngram_range=tuple(m["ngram_range"]),
                    min_df=m["min_df"],
                    max_features=m["max_features"],
                ),
                LogisticRegression(C=m["C"], max_iter=m["max_iter"], random_state=seed),
            )

            start = time.time()
            model.fit(x_train, train["label"])
            train_time = time.time() - start
            if train_time > cfg["limits"]["timeout_seconds"]:
                logging.warning("Training took %.1f s - over the time limit", train_time)

            top_share, vocab_share = source_specific_share(model, cl["source_words"], exp["top_n"])

            for test_set, x, y in [("isot_test", x_test, test["label"]),
                                   ("liar", x_liar, liar["label"])]:
                pred = model.predict(x)
                row = {
                    "seed": seed,
                    "text_version": version,
                    "test_set": test_set,
                    "accuracy": round(accuracy_score(y, pred), 4),
                    "f1": round(f1_score(y, pred, average="macro"), 4),
                    "top_n_source_share": round(top_share, 4),
                    "vocab_source_share": round(vocab_share, 4),
                    "train_time_s": round(train_time, 3),
                }
                results.append(row)
                logging.info("seed=%s version=%s test=%s accuracy=%.4f f1=%.4f",
                             seed, version, test_set, row["accuracy"], row["f1"])

    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    summary = df.groupby(["text_version", "test_set"])[["accuracy", "f1"]].mean().round(4)
    logging.info("Mean over seeds:\n%s", summary)

    isot_f1 = df[df["test_set"] == "isot_test"]["f1"].mean()
    if isot_f1 < cfg["limits"]["min_isot_f1"]:
        logging.warning("ISOT test F1 %.4f is below the limit %.2f",
                        isot_f1, cfg["limits"]["min_isot_f1"])

    logging.info("Results saved to %s and %s", csv_file, log_file)


if __name__ == "__main__":
    main()
