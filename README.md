# Sentiment Analysis of Financial News Tweets Using DistilBERT

## Overview
This project fine-tunes the DistilBERT model on annotated financial news tweets to classify sentiments into **Bullish**, **Bearish**, and **Neutral**. The fine-tuned model is evaluated on a test set, and an inference application is created for real-time predictions.

## Key Features
- **Fine-Tuning**: Used Hugging Face's `transformers` library to fine-tune DistilBERT on financial tweets.
- **Custom Dataset**: Financial news tweets labeled as Bullish, Bearish, or Neutral.
- **Interactive App**: An inference application for real-time sentiment predictions on new tweets.
- **Performance Metrics**:
  - Accuracy: 87.43%
  - F1-Score: 87.08%

## Technologies Used
- Python
- Hugging Face Transformers
- PyTorch
- Matplotlib, Seaborn
- PyQt5

## Dataset
The dataset consists of financial news tweets, pre-labeled with sentiment:
- **Classes**:
  - `Bullish`
  - `Bearish`
  - `Neutral`
- **Data Split**:
  - Training: 70%
  - Validation: 20%
  - Test: 10%

## Results
- **Best Model**: DistilBERT fine-tuned on financial news tweets.
- **Performance Metrics**:
  - Accuracy: 87.43%
  - F1-Score: 87.08%

## File Descriptions
- **`fine_tune_distilbert.ipynb`**: Colab notebook to fine-tune DistilBERT and evaluate the model on test data.
- **`inference_gui.py`**: application for real-time sentiment predictions.
## Instructions to Run
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Financial-News-Sentiment-Analysis.git
cd Financial-News-Sentiment-Analysis
