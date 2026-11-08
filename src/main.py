import argparse
import json
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

    # if some label is not in liar_label_map, stop and show which one
    unknown = data[data["label"].isna()]["label_6"].unique()
    if len(unknown) > 0:
        raise ValueError(f"LIAR has labels that are not in liar_label_map: {list(unknown)}")

    data["label"] = data["label"].astype(int)
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
    # values are NOT rounded, we need full precision for the statistics later
    precision = precision_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    recall = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "precision_real": float(precision[0]),
        "recall_real": float(recall[0]),
        "precision_fake": float(precision[1]),
        "recall_fake": float(recall[1]),
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
    raw_file = cfg["output"]["raw_file"]
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

    exp, m, cl = cfg["experiment"], cfg["model"], cfg["cleaning"]

    # now we clean texts only one time, before the loops
    isot["clean_text"] = isot["text"].apply(clean_text, patterns=cl["patterns"])
    liar["clean_text"] = liar["text"].apply(clean_text, patterns=cl["patterns"])
    text_column = {"raw": "text", "clean": "clean_text"}

    # sanity check: majority-class classifier on LIAR
    majority = liar["label"].mode()[0]
    majority_scores = scores(liar["label"], [majority] * len(liar))
    logging.info("Majority-class baseline on LIAR: accuracy=%.4f f1=%.4f",
                 majority_scores["accuracy"], majority_scores["f1"])

    records = [{"record_type": "metrics", "phase": "benchmark", "seed": None,
                "model": "majority_class", "text_version": None, "test_set": "liar",
                **majority_scores}]

    # we do warm up firsly, they're marked as "warmup", the real ones as "benchmark"
    runs = [(s, "warmup") for s in exp["warmup_seeds"]] + [(s, "benchmark") for s in exp["seeds"]]

    for seed, phase in runs:
        train, test = train_test_split(
            isot, test_size=exp["test_size"], stratify=isot["label"], random_state=seed
        )
        overlap = len(set(train["text"]) & set(test["text"]))
        if overlap > 0:
            logging.warning("Data leakage: %d texts are both in ISOT train and test", overlap)

        seed_records = []
        isot_f1 = {}
        for version in exp["text_versions"]:
            col = text_column[version]
            x_train, x_test, x_liar = train[col], test[col], liar[col]

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

            # H2: share of source-specific features in the top-N for every N.
            shares = {}
            for n in exp["top_n"]:
                top_share, vocab_share = source_specific_share(model, cl["source_words"], n)
                shares[f"top{n}_source_share"] = top_share
            shares["vocab_source_share"] = vocab_share
            seed_records.append({"record_type": "shares", "phase": phase, "seed": seed,
                                 "model": "tfidf_logreg", "text_version": version,
                                 "test_set": None, **shares})

            for test_set, x, y in [("isot_test", x_test, test["label"]),
                                   ("liar", x_liar, liar["label"])]:
                row = {"record_type": "metrics", "phase": phase, "seed": seed,
                       "model": "tfidf_logreg", "text_version": version,
                       "test_set": test_set, **scores(y, model.predict(x)),
                       "train_time_s": train_time}
                seed_records.append(row)
                logging.info("phase=%s seed=%s version=%s test=%s accuracy=%.4f f1=%.4f",
                             phase, seed, version, test_set, row["accuracy"], row["f1"])
                if test_set == "isot_test":
                    isot_f1[version] = row["f1"]

        # guardrail: cleaning must not lower ISOT test F1 by more than the limit.
        guardrail = {"train_test_overlap": overlap}
        if "raw" in isot_f1 and "clean" in isot_f1:
            drop = isot_f1["raw"] - isot_f1["clean"]
            guardrail["isot_f1_drop"] = drop
            guardrail["guardrail_ok"] = bool(drop <= cfg["limits"]["max_isot_f1_drop"])
            if not guardrail["guardrail_ok"]:
                logging.warning("seed=%s: ISOT test F1 dropped by %.4f after cleaning "
                                "(limit %.2f)", seed, drop, cfg["limits"]["max_isot_f1_drop"])
        for record in seed_records:
            record.update(guardrail)
        records.extend(seed_records)

    # save raw results: one JSON line = one record
    os.makedirs(os.path.dirname(raw_file), exist_ok=True)
    with open(raw_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # print summary (only benchmark runs, warm-up is not counted)
    df = pd.DataFrame(records)
    bench = df[(df["phase"] == "benchmark") & (df["record_type"] == "metrics")
               & (df["model"] == "tfidf_logreg")]
    summary = bench.groupby(["text_version", "test_set"])[["accuracy", "f1"]].agg(["mean", "std"])
    logging.info("Mean and SD over seeds:\n%s", summary.round(4))
    logging.info("Results saved to %s and %s", raw_file, log_file)


if __name__ == "__main__":
    main()