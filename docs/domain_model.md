# Domain model

All values are provided initially in config.yaml.

## 1 Data 
We solve binary classification problem

- A text (title + body of a news) is *d*
- A label is $y \in \{0, 1\}$. $y = 0$ is real and $y = 1$ is fake.
- Source datasets are: ISOT, $\mathcal{D}_{\mathrm{ISOT}}$. $\mathcal{D}_{\mathrm{ISOT}}$ is split to 80-20 training parts, stratified by label.
- Unseen datasets are: LIAR, $\mathcal{D}_{\mathrm{LIAR}}$

LIAR has 6 labels that divided to two classes:
- Real: true, mostly-true, half-true (0)
- Fake: pants-fire, false, barely-true (1)

## 2 Text representation

Every text *d* is turned into vector with 50000 features (including words and words pairs)

For each feature t in text *d*:
$$
\mathrm{tfidf}(t, d) = \mathrm{tf}(t, d) \cdot \mathrm{idf}(t), \qquad
\mathrm{idf}(t) = \ln \frac{1 + n}{1 + \mathrm{df}(t)} + 1 ,
$$

Where: 
- $\mathrm{tf}(t, d)$ is how many times t appears in *d*.
- $\mathrm{df}(t)$ is the number of training text that contains t. 
- $n$ is the number of training texts.

Then we apply L2 normalisation to each vector

## 3 Model and loss function

Our classifier is Logistic Regression with parameters *w* and bias *b*

The probability of that a text is fake:
$$
p(\mathbf{x}) = \sigma(\mathbf{w}^{\top}\mathbf{x} + b), \qquad
\sigma(z) = \frac{1}{1 + e^{-z}} .
$$

Loss function L:
$$
L(\mathbf{w}, b) =
\underbrace{\frac{1}{2}\,\mathbf{w}^{\top}\mathbf{w}}_{\text{L2 regularization}}
\;+\;
C \sum_{i=1}^{n} \underbrace{\Big( - y_i \ln p_i - (1 - y_i)\ln (1 - p_i) \Big)}_{\text{cross-entropy (log loss)}},
\qquad p_i = p(\mathbf{x}_i).
$$

Trained parameters:
$$(\mathbf{w}^{*}, b^{*}) = \arg\min_{\mathbf{w},\, b} \; L(\mathbf{w}, b).$$

C = 1.0 is regularization strength parameter.

## 4 Target metrics
For every class (real, fake), we count True-Positive, False-Positive and False-Negative on a test set.

Using TP, FP, FN we may find precision, recall and F1 metrics:
$$
\mathrm{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \qquad
\mathrm{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \qquad
F1_c = \frac{2\,\mathrm{Precision}_c\,\mathrm{Recall}_c}{\mathrm{Precision}_c + \mathrm{Recall}_c}.
$$

Macro F1:
$$F1 = \frac{F1_{\mathrm{real}} + F1_{\mathrm{fake}}}{2}.$$

Accuracy:
$$\mathrm{Accuracy} = \frac{TP_{\mathrm{real}} + TP_{\mathrm{fake}}}{\text{number of texts}} .$$

Performance drop:
$$\Delta_{\mathrm{drop}}(s) = F1_{\mathrm{ISOT\ test}}(s) - F1_{\mathrm{LIAR}}(s).$$

Calculation of cleaning effect:
$$\Delta F1(s) = F1^{\mathrm{clean}}_{\mathrm{LIAR}}(s) - F1^{\mathrm{raw}}_{\mathrm{LIAR}}(s).$$

Guardrail:
$$F1^{\mathrm{raw}}_{\mathrm{ISOT\ test}}(s) - F1^{\mathrm{clean}}_{\mathrm{ISOT\ test}}(s) \le 0.02 .$$

## 5 Source-specific features (H2)

Imagine *V* is a vocabulary of all features of trained model.
W is list of *source_words* from *config.yaml*: 'reuters', 'featured' and so on. 

A feature $f \in V$ is *source-specific* if at least one of its words is in *W*:
$$
\mathrm{ss}(f) =
\begin{cases}
1, & \text{if some word of } f \text{ belongs to } W, \\
0, & \text{otherwise.}
\end{cases}
$$

Features ranked by the absolute value of their weight $\lvert w_j \rvert$.
$\mathrm{Top}_N$ is the *N* features with largest $\lvert w_j \rvert$. $N \in \{100, 500, 1000\}$.<br>
Then:
$$
S_N = \frac{1}{N} \sum_{f \in \mathrm{Top}_N} \mathrm{ss}(f), \qquad
S_V = \frac{1}{\lvert V \rvert} \sum_{f \in V} \mathrm{ss}(f).
$$

If *source-specific* features weren't special, then number $k = N \cdot S_N$ would be likely:
$$k \sim \mathrm{Binomial}(N, S_V).$$

## 6 Cleaning
Cleaning function cleaning(d) removes all patterns from *cleaning.patterns* in *configs/config.yaml*.
It's been applied with same rules to ISOT and to LIAR.

## 7 Statistics

- Used seeds: $\mathcal{S} = \{13, 21, 42, 87, 123, 7, 99, 256, 314, 777\}$. Warm-up saved separately
- For metric $\mu(s)$:
$$
\bar{\mu} = \frac{1}{10} \sum_{s \in \mathcal{S}} \mu(s), \qquad
\mathrm{SD} = \sqrt{\frac{1}{9} \sum_{s \in \mathcal{S}} \big(\mu(s) - \bar{\mu}\big)^2 },
$$
- Bootstrap 95% CI of mean: 10000 samples of size 10 from $\mu(s)$ with replacement, then compute mean, take 2.5% and 97.5% percentiles of these means. 
- *H1 (drop on unseen data).* One-sided paired $t$-test of $\Delta_{\mathrm{drop}}(s) > 0$.
  Supported if $p < 0.05$ and the mean drop is larger than the SD over seeds.
- *H2 (source-specific features).* One-sided binomial test from Section 5 for every seed and every $N$:
  $\mathrm{H}_0: S_N \le S_V$, $\mathrm{H}_1: S_N > S_V$.
  Supported if the largest $p$-value over the seeds is below 0.05.
- *H3 (effect of cleaning).* Bootstrap 95% CI of the mean $\Delta F1(s)$.
  Supported if the CI does not include 0 and the mean $\Delta F1$ is larger than the SD over seeds.

## 8 Notation table

| Symbol | Meaning |
|---|---|
| $d$ | a text (title + body) |
| $y \in \{0, 1\}$ | true label: 0 = real, 1 = fake |
| $\mathcal{D}_{\mathrm{ISOT}}$, $\mathcal{D}_{\mathrm{LIAR}}$ | ISOT (training source) and LIAR (unseen) datasets |
| $n$ | number of training texts |
| $V$ | vocabulary: all features (words and word pairs) |
| $\mathbf{x}(d)$ | TF-IDF vector of text $d$ |
| $\mathbf{w}$, $b$ | model weights and bias |
| $\sigma(z)$ | sigmoid function |
| $p(\mathbf{x})$ | predicted probability that a text is fake |
| $L(\mathbf{w}, b)$ | loss function |
| $C$ | regularization strength, $C = 1.0$ |
| $c$ | a class: real or fake |
| $TP_c$, $FP_c$, $FN_c$ | true positives, false positives, false negatives of class $c$ |
| $F1$ | macro-averaged F1 (primary metric) |
| $s$, $\mathcal{S}$ | a random seed and the set of 10 seeds |
| $\mu(s)$ | value of a metric for seed $s$ |
| $\bar{\mu}$, $\mathrm{SD}$ | mean and standard deviation over seeds |
| $\Delta_{\mathrm{drop}}(s)$ | F1 drop: ISOT test minus LIAR (H1) |
| $\Delta F1(s)$ | F1 change on LIAR after cleaning (H3) |
| $W$ | list of source words from config |
| $\mathrm{ss}(f)$ | 1 if feature $f$ is source-specific, else 0 |
| $N$, $\mathrm{Top}_N$ | top list size and the $N$ features with the largest absolute weight |
| $S_N$, $S_V$ | share of source-specific features in $\mathrm{Top}_N$ and in the whole vocabulary |
| $cleanin(d)$ | cleaning function |