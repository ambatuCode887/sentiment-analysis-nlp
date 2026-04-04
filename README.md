#Amazon Food Reviews - Sentiment Analysis

An NLP project that performs sentiment analysis on Amazon food reviews using both traditional machine learning (scikit-learn) and modern deep learning (HuggingFace Transformers).

##Project Overview
This is my third Python project, built as part of my journey towards becoming an NLP/AI Engineer.
The goal is to classify Amazon food reviews as Positive or Negative using two different approaches and compare their performance.

##Dataset
- Source: [Kaggle Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Original size: 525,814 reviews
- Used: 10,000 sampled reviews
- Features used: Review Text, Star Score (converted to sentiment)

##Label Creation
| Score | Sentiment |
|---|---|
| 4-5 stars | Positive (1) |
| 1-2 stars | Negative (0) |
| 3 stars | Dropped (ambiguous) |

##What I Did
1. **Data Loading** — loaded and explored 525,814 Amazon food reviews
2. **Sampling** — sampled 10,000 rows for manageable training
3. **Text Cleaning** — lowercased, removed symbols, removed stopwords
4. **TF-IDF Vectorization** — converted text to numbers (top 5000 features, unigrams + bigrams)
5. **Model Training** — trained Logistic Regression on TF-IDF features
6. **HuggingFace** — tested pretrained DistilBERT model on raw text

##Results
| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression + TF-IDF | 92.82% | Trained on specific dataset |
| HuggingFace DistilBERT | ~78-83% | General purpose pretrained model |
| Ensemble | TBD | Combination of both |

##Key Findings
- Logistic Regression outperformed HuggingFace on this specific dataset
- Class imbalance (85% positive reviews) boosted LR accuracy
- Feature engineering helps traditional ML but HURTS HuggingFace/BERT
- More test samples give more honest accuracy scores
- Specific trained models beat general pretrained models on narrow tasks

##Key Concepts Learned
- TF-IDF vectorization and ngram ranges
- Difference between traditional ML and transformer based NLP
- Why feature engineering hurts deep learning models
- Class imbalance and its effect on accuracy

##Libraries Used
- pandas, numpy
- scikit-learn
- nltk
- transformers (HuggingFace)
- matplotlib, seaborn

##How to Run
1. Clone this repo
2. Download dataset from Kaggle and place in a separate folder
3. Update the CSV path in the notebook
4. Install dependencies:
```bash
pip install pandas numpy scikit-learn nltk transformers torch matplotlib seaborn
```
5. Open `sentiment.ipynb` in VSCode
6. Run all cells in order

## 📚 What I Learned
- Full NLP pipeline from raw text to trained model
- How transformers and BERT work vs traditional ML
- Why bigger models don't always win
- How to handle large datasets efficiently
- Ensemble methods in NLP
