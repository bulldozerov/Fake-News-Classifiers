import argparse
import logging
import os
import re
import time
from datetime import datetime

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def load_isot(true_path, fake_path):
    real = pd.read_csv(true_path)
    fake = pd.read_csv(fake_path)
    real["label"] = 0
    fake["label"] = 1
    data = pd.concat([real, fake], ignore_index=True)
    data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")
    data["text"] = data["text"].str.strip()

    data = data[data["text"] != ""]
    data = data.drop_duplicates(subset="text")
    return data[["text", "label"]].reset_index(drop=True)


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


def source_specific_share(model, source_words, n):
    words = model.named_steps["tfidfvectorizer"].get_feature_names_out()
    weights = model.named_steps["logisticregression"].coef_[0]
    top = sorted(zip(words, weights), key=lambda x: abs(x[1]), reverse=True)[:n]
    top_share = sum(is_source_specific(w, source_words) for w, _ in top) / len(top)
    vocab_share = sum(is_source_specific(w, source_words) for w in words) / len(words)
    return top_share, vocab_share


def scores(y, pred):
    precision = precision_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    recall = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "f1": round(f1_score(y, pred, average="macro", zero_division=0), 4),
        "precision_real": round(precision[0], 4),
        "recall_real": round(recall[0], 4),
        "precision_fake": round(precision[1], 4),
        "recall_fake": round(recall[1], 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    # load data
    isot = load_isot(cfg["data"]["isot_true"], cfg["data"]["isot_fake"])
    liar = load_liar(cfg["data"]["liar"], cfg["data"]["liar_label_map"])
    logging.info("ISOT: %d rows after removing duplicates, LIAR: %d rows", len(isot), len(liar))
    logging.info("LIAR class balance: real=%.3f fake=%.3f",
                 (liar["label"] == 0).mean(), (liar["label"] == 1).mean())

    # sanity check: majority-class classifier on LIAR
    majority = liar["label"].mode()[0]
    majority_scores = scores(liar["label"], [majority] * len(liar))
    logging.info("Majority-class baseline on LIAR: accuracy=%.4f f1=%.4f",
                 majority_scores["accuracy"], majority_scores["f1"])

    exp, m, cl = cfg["experiment"], cfg["model"], cfg["cleaning"]
    results = [{"seed": "-", "model": "majority_class", "text_version": "-",
                "test_set": "liar", **majority_scores}]

    for seed in exp["seeds"]:
        train, test = train_test_split(
            isot, test_size=exp["test_size"], stratify=isot["label"], random_state=seed
        )
        overlap = len(set(train["text"]) & set(test["text"]))
        if overlap > 0:
            logging.warning("Data leakage: %d texts are both in ISOT train and test", overlap)

        isot_f1 = {}
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
                LogisticRegression(C=m["C"], solver=m["solver"], max_iter=m["max_iter"],
                                   random_state=seed),
            )

            start = time.time()
            model.fit(x_train, train["label"])
            train_time = time.time() - start
            if train_time > cfg["limits"]["timeout_seconds"]:
                logging.warning("Training took %.1f s - over the time limit", train_time)

            # H2: share of source-specific features in the top-N for every N
            shares = {}
            for n in exp["top_n"]:
                top_share, vocab_share = source_specific_share(model, cl["source_words"], n)
                shares[f"top{n}_source_share"] = round(top_share, 4)
            shares["vocab_source_share"] = round(vocab_share, 4)

            for test_set, x, y in [("isot_test", x_test, test["label"]),
                                   ("liar", x_liar, liar["label"])]:
                row = {"seed": seed, "model": "tfidf_logreg", "text_version": version,
                       "test_set": test_set, **scores(y, model.predict(x)), **shares,
                       "train_time_s": round(train_time, 3)}
                results.append(row)
                logging.info("seed=%s version=%s test=%s accuracy=%.4f f1=%.4f",
                             seed, version, test_set, row["accuracy"], row["f1"])
                if test_set == "isot_test":
                    isot_f1[version] = row["f1"]

        # guardrail: cleaning must not lower ISOT test F1 by more than the limit
        if "raw" in isot_f1 and "clean" in isot_f1:
            drop = isot_f1["raw"] - isot_f1["clean"]
            if drop > cfg["limits"]["max_isot_f1_drop"]:
                logging.warning("seed=%s: ISOT test F1 dropped by %.4f after cleaning "
                                "(limit %.2f)", seed, drop, cfg["limits"]["max_isot_f1_drop"])

    # save and print summary
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    model_rows = df[df["model"] == "tfidf_logreg"]
    summary = model_rows.groupby(["text_version", "test_set"])[["accuracy", "f1"]].agg(["mean", "std"])
    logging.info("Mean and SD over seeds:\n%s", summary.round(4))
    logging.info("Results saved to %s and %s", csv_file, log_file)


if __name__ == "__main__":
    main()
