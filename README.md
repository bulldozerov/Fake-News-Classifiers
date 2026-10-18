# Which Text Features Cause Performance Degradation of Fake News Classifiers on Unseen Data?

**Authors:** Ashim Burabayev, Ibrakhim Olzhabai (Narxoz University)

## Overview

Fake news classifiers usually work much worse on data from another source than on the
data they were trained on. In this project we train a TF-IDF + Logistic Regression
classifier on the ISOT dataset and test it on the LIAR dataset. We also check whether
source-specific features, such as agency names, image credits and links, cause the
drop in performance.

## Research Question

Which text features cause performance degradation of fake news classifiers on unseen data?

- **RQ1:** How much do F1 and accuracy drop when a model trained on ISOT is tested on LIAR?
- **RQ2:** How much do the highest-weighted features rely on source-specific clues?
- **RQ3:** How does removing source-specific features affect performance on LIAR?

## Repository Structure

```
Fake-News-Classifiers/
├── configs/
│   └── config.yaml
├── data/
│   └── sample/
│       ├── isot_true_sample.csv
│       ├── isot_fake_sample.csv
│       └── liar_sample.tsv
├── notebooks/
├── src/
│   └── main.py
├── Dockerfile
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Planned Benchmarks

| ID | Hypothesis | Train data | Test data | Independent variable | Metrics |
|----|------------|------------|-----------|----------------------|---------|
| B0 | – | – | LIAR | – | majority-class baseline: F1, accuracy |
| B1 | RQ1 | ISOT train (raw text) | ISOT test vs. LIAR | test data source | F1, accuracy |
| B2 | RQ2 | ISOT train (raw text) | – | feature rank by weight (top-100, 500, 1000) | share of source-specific features in top-N vs. vocabulary |
| B3 | RQ3 | ISOT train (raw vs. clean text) | ISOT test, LIAR | text version | F1, accuracy |

Each benchmark is repeated with 10 random seeds.

## Quickstart

```bash
git clone https://github.com/bulldozerov/Fake-News-Classifiers.git && cd Fake-News-Classifiers
docker build -t fake-news-benchmark .
docker run --rm -v "$(pwd)/results:/app/results" fake-news-benchmark
```

The test run uses `data/sample/` and writes `results/run_<time>.log` and
`results/metrics_<time>.csv`.

Without Docker:

```bash
pip install -r requirements.txt
python src/main.py --config configs/config.yaml
```

## Data

- **ISOT Fake News Dataset**
  <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>
- **LIAR**
  <https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset>


## License

MIT
