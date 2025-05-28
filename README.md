# Movie Review Sentiment Classification
## Egor Bodrov, Onore Gleb, Evgeniy Dubskiy
May 2025

Abstract
This project addresses the problem of automatic sentiment classification of movie reviews using modern NLP models. We compare several transformer-based approaches on the Kaggle dataset [dsaa-6100-movie-review-sentiment-classification](https://www.kaggle.com/competitions/dsaa-6100-movie-review-sentiment-classification/overview). The goal is to achieve high accuracy in binary sentiment classification (positive/negative) and analyze the impact of model choice and preprocessing.

Project code: https://github.com/EgorBodrov/ods-nlp.git

# 1 Introduction
Sentiment analysis of movie reviews is a classic NLP task with practical applications in recommendation systems, social media monitoring, and customer feedback analysis. The challenge lies in handling informal language, sarcasm, and short texts. We focus on leveraging recent advances in transformer architectures (DistilBERT, DeBERTa v3) and compare their performance on a real-world dataset.

## 1.1 Team
Egor Bodrov - ...
Onore Gleb - ...
Evgeniy Dubskiy - ...

# 2 Related Work
- Pang & Lee (2008): Early work on sentiment analysis using bag-of-words and SVMs.
- Howard & Ruder (2018): ULMFiT for transfer learning in text classification.
- Devlin et al. (2019): BERT and its variants for state-of-the-art text classification.
- He et al. (2021): DeBERTa v3, improved transformer architecture for NLP tasks.

# 3 Model Description
We evaluated the following models:
- **DistilBERT (pretrained):** `sarahai/movie-sentiment-analysis` pipeline, no fine-tuning.
- **DistilBERT (fine-tuned):** Fine-tuned on the provided movie review dataset.
- **DistilBERT (SST-2):** Pretrained on SST-2, used as a zero-shot baseline.
- **DistilBERT (Amazon):** Pretrained on Amazon reviews, used as a zero-shot baseline.
- **DeBERTa v3 (pretrained):** `microsoft/deberta-v3-base` and `dfurman/deberta-v3-base-imdb`, fine-tuned on the dataset.

All models are used for binary classification (positive/negative). Fine-tuning was performed using HuggingFace Transformers and PyTorch.

# 4 Dataset

## 4.1 Source and Access
- **Dataset:** [Kaggle dsaa-6100-movie-review-sentiment-classification](https://www.kaggle.com/competitions/dsaa-6100-movie-review-sentiment-classification/overview)
- **License:** For academic use, available via Kaggle competition page.
- **Files:**
  - `movie_reviews.csv` (train): columns `Id`, `text`, `label` (0=negative, 1=positive)
  - `test_data.csv` (test): columns `Id`, `text`

## 4.2 Statistics & EDA
- **Train size:** 40,000 reviews
- **Test size:** 10,000 reviews
- **Class balance:**
  - Positive (1): 50.1%
  - Negative (0): 49.9%
- **Text characteristics:**
  - Length distribution: 100-500 characters (majority)
  - Average words per review: ~100 words
  - Duplicate reviews present: ~2.5% of total reviews
- See `experiments/EDA.ipynb` for detailed visualizations:
  - Class distribution plots
  - Text length histograms
  - Word clouds for frequent terms
  - Comparison of positive vs negative review lengths

## 4.3 Preprocessing
Our preprocessing pipeline includes:
1. HTML tag removal (e.g., `<br>` tags)
2. Text normalization:
   - Lowercasing
   - Extra space removal
   - Special character handling
3. NLTK-based processing:
   - Tokenization
   - Stemming (Porter Stemmer)
   - Lemmatization (WordNet)

# 5 Experiments

## 5.1 Metrics
- **Primary metric:** Accuracy
- **Rationale:** Dataset is balanced, accuracy is the competition metric

## 5.2 Experiment Setup
- **Data split:** 90% train / 10% validation (stratified)
- **Training parameters:**
  - Batch size: 16
  - Max sequence length: 256 tokens
  - Optimizer: AdamW (lr=2e-5)
  - Epochs: 3-4 with early stopping
  - Loss: Cross-entropy
- **Hardware:** GPU (CUDA if available)
- **Model selection:** Best checkpoint by validation accuracy

## 5.3 Models and Training
1. **DeBERTa v3 IMDB (dfurman/deberta-v3-base-imdb)**
   - Pre-trained on IMDB reviews
   - Fine-tuned on our dataset
   - Implementation: `experiments/deberta-v3-base-imdb.ipynb`

2. **DistilBERT Fine-tuned**
   - Base: `distilbert-base-uncased`
   - Custom fine-tuning on movie reviews
   - Implementation: `experiments/distilbert_finetuned.ipynb`

3. **Zero-shot Baselines**
   - DistilBERT (no fine-tuning): `sarahai/movie-sentiment-analysis`
   - DistilBERT SST-2: Pre-trained on Stanford Sentiment Treebank
   - DistilBERT Amazon: Pre-trained on Amazon reviews

# 6 Results
| Model                              | Validation Accuracy |
|-------------------------------------|--------------------|
| DistilBERT (pretrained, no FT)      | ~0.85              |
| DistilBERT (fine-tuned)             | ~0.89              |
| DistilBERT (SST-2, zero-shot)       | ~0.83              |
| DistilBERT (Amazon, zero-shot)      | ~0.81              |
| DeBERTa v3 (fine-tuned, IMDB)       | ~0.90              |
| DeBERTa v3 (fine-tuned, base)       | ~0.89              |

*See experiment notebooks for detailed logs and plots.*

# 7 Conclusion
We compared several transformer-based models for movie review sentiment classification. Fine-tuned DeBERTa v3 and DistilBERT models outperform zero-shot and out-of-domain baselines, achieving up to 90% accuracy. Preprocessing and careful validation are crucial for robust results. Future work may include ensembling, data augmentation, and error analysis.

# References
- [Kaggle Competition Page](https://www.kaggle.com/competitions/dsaa-6100-movie-review-sentiment-classification/overview)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2019
- He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention", 2021
- Howard & Ruder, "Universal Language Model Fine-tuning for Text Classification", 2018
- Pang & Lee, "Opinion Mining and Sentiment Analysis", 2008