# Performance degradation of Fake News Classifiers on Unseen Data

**Authors:** Ashim Burabayev, Ibrakhim Olzhabai (Narxoz University)

## Overview

Classifires made for detecting fake news usually works very badly at the data from another source. In this project we train TF-IDF + Logistic regression 
classifier based on ISOT dataset, and test it on LIAR dataset. Also we check whether specific features like agency names, image credits, link and other, 
cause the drop in perfomance.

## Research Question

Which Text Features Cause Performance Degradation of Fake News Classifiers on Unseen Data?

- **RQ1:** How much do F1 and accuracy drop when a model trained on ISOT is tested on LIAR?
- **RQ2:** How much do the high-weighted features rely on source-specific clues?
- **RQ3:** How does removing source-specific features affect performance on LIAR?

## Repository Structure

fake-news-shortcut-audit/
  +-- configs/
    |   +-- config.yaml            
  +-- data/
    |   +-- sample/                
    |       +-- isot_true_sample.csv
    |       +-- isot_fake_sample.csv
    |       +-- liar_sample.tsv
  +-- notebooks/                 
  +-- src/
    |   +-- main.py                
  +-- Dockerfile
  +-- requirements.txt         
  +-- .gitignore
  +-- LICENSE
  +-- README.md


## Planned Benchmarks

| ID | Hypothesis | Train data                        | Test data          | Independent variable            | Metrics                           |
| B1 | RQ1        | ISOT train (raw text)             | ISOT test vs. LIAR | test data source                | F1, accuracy                      |
| B2 | RQ2        | ISOT train (raw text)             | –                  | feature rank by weight (top-20) | share of source-specific features |
| B3 | RQ3        | ISOT train (raw vs. clean text)   | LIAR               | text version                    | F1, accuracy                      |

Each benchmark is repeated with 5 random seeds (13, 21, 42, 87, 123).

## Quickstart

```bash
git clone https://github.com/<username>/fake-news-shortcut-audit.git && cd fake-news-shortcut-audit
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
  <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download>
- **LIAR** (Wang, 2017), used only for testing:
  <https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset?select=README>

## License

MIT
