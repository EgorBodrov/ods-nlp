# NLP Course Template
## Egor Bodrov, Onore Gleb, Evgeniy Dubskiy
May 2025
Abstract
This document will provide you with guidelines for your project final
report. You will learn how to structure the report and present your results.
Use this field for the short description of your work. Please provide a
link to your project code right here: https://github.com/EgorBodrov/ods-nlp.git
# 1 Introduction
First of all, you will need to write the whole report in English, with a few
exceptions mentioned below. This section is devoted to a problem motivation.
You should answer the question of why the problem you were working on is
important. Also, you should describe what is unique in your approach to this
problem, what are the differences to other approaches.
## 1.1 Team
Egor Bodrov - ...
Onore Gleb - ...
Evgeniy Dubskiy - ...

# 2 Related Work

# 3 Model Description

# 4 Dataset

Вот аккуратно оформленный Markdown `.md` вариант описания датасета **RCV1**, аналогичный по стилю твоему примеру с PAWS:

---

## 4 Dataset

In this section, we provide a detailed description of the **Reuters Corpus Volume I (RCV1)** dataset, including its origin, licensing conditions, acquisition methods, and key statistics. RCV1 is one of the largest publicly available corpora for text categorization tasks and is widely used in machine learning and information retrieval research.

RCV1 was first introduced by Lewis *et al.* \[2004], where document encoding principles, taxonomy structures, and quality control procedures were presented. The corpus contains over **800,000** English-language news stories manually annotated along three hierarchical taxonomies: **topics**, **industries**, and **regions**.
---

### Dataset Access

**Official source:** Access can be requested via the [NIST TREC project](https://trec.nist.gov/data/reuters/reuters.html) under “Reuters Corpora — RCV1”.

**Using scikit-learn:** The built-in `fetch_rcv1` loader automatically downloads and caches the dataset in `~/scikit_learn_data/rcv1`:

```python
from sklearn.datasets import fetch_rcv1
rcv1 = fetch_rcv1(subset='all', download_if_missing=True)
```

More details can be found in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html).

---

### Data Format

* `data`: a **CSR sparse matrix** of shape `(804,414 × 47,236)` with **log TF–IDF values** normalized by cosine.
* `target`: a **multi-label sparse matrix**:

  * `(804,414 × 103)` for topics
  * `(804,414 × 369)` for industries
  * `(804,414 × 54)` for regions
* `sample_id`: an array of unique string identifiers for each article.

---

### Preprocessing and Splits

RCV1 comes with a **chronological split (LYRL2004)**:

| Split | #Articles |
| ----- | --------- |
| Train | 23,149    |
| Test  | 781,265   |

This split is reflected in `fetch_rcv1(subset='train')` and `fetch_rcv1(subset='test')`. There is no official **validation set**, but many studies reserve **5–10%** of the training data for validation.

---

### Dataset Statistics

#### Table 1. Basic statistics of the RCV1 corpus.

| Split | Articles | Vocabulary Size          | Non-zero Values (%)     | Description                            |
| ----- | -------- | ------------------------ | ----------------------- | -------------------------------------- |
| Train | 23,149   | \multirow{2}{\*}{47,236} | \multirow{2}{\*}{0.16%} | Articles before split (initial subset) |
| Test  | 781,265  |                          |                         | Articles after split (remainder)       |
| Total | 804,414  |                          |                         | Entire corpus                          |

> **Note:** The non-zero percentage reflects the proportion of non-zero entries in the TF–IDF matrix.

---

### Taxonomy Label Counts

| Taxonomy   | #Labels |
| ---------- | ------- |
| Topics     | 103     |
| Industries | 369     |
| Regions    | 54      |

These hierarchies are described in Lewis *et al.* \[2004] and used for **multi-label classification** tasks.


---

Thus, **RCV1** is a well-documented and robust corpus for **text categorization** research, accessible through both official channels and popular libraries like **scikit-learn**.


# 5 Experiments
This section should include several subsections.
## 5.1 Metrics
First of all, you should describe the metric(s) you were using to evaluate your
approach. Most likely a metric description will include a formula.
## 5.2 Experiment Setup
Secondly, you need to describe the design of your experiment, e.g. how many
runs there were, how the data split was done. The important details of your
model, like hyper-parameters used in the experiments, and so on.
## 5.3 Baselines
Another important feature is that you could provide here the description of
some simple approaches for your problem, like logistic regression over TF-IDF
embedding for text classification. The baselines are needed is there is no previous
art on the problem you are presenting.

# 6 Results

# 7 Conclusion
In this section, you need to describe all the work in short: what you have done
and what has been achieved. E.g. you have collected a dataset, made a markup
for it and developed a model showing the best results compared to other models.
References