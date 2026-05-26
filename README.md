# Indonesian Political Fake News Classifier Using Support Vector Machine (SVM) and Explainable AI (LIME)

## 📌 1. Project Overview
This repository contains the complete source code, experimental notebooks, and documentation for my Undergraduate Thesis in Informatics Engineering at Institut Teknologi PLN. The primary objective of this research is to tackle the massive spread of misinformation and hoaxes within the Indonesian political landscape.

To achieve this, the project implements a robust machine learning pipeline powered by the **Support Vector Machine (SVM)** algorithm. Crucially, the system integrates an **Explainable Artificial Intelligence (XAI)** framework via **LIME (Local Interpretable Model-agnostic Explanations)** to dismantle the "black-box" nature of the model, providing full linguistic transparency behind every classification decision.

## 🛠️ 2. Tech Stack & Libraries
* **Language:** Python 3.x
* **Core Machine Learning:** Scikit-Learn (SVM Classifier, Grid Search, Multi-kernel Optimization)
* **Text Processing & Feature Engineering:** NLTK, Sastrawi, TfidfVectorizer
* **Explainable AI Module:** LimeTextExplainer (LIME Library)
* **Data Visualization:** Matplotlib, Seaborn

## ⚙️ 3. Methodology & Experimental Pipeline
The research is structured around a rigorous data engineering and modeling pipeline:
1. **Text Preprocessing:** Comprehensive raw text cleaning (removing URLs, mentions, hashtags, digits, and special characters), case folding, tokenization, slang normalization, stopword removal, and word stemming using the *Sastrawi Stemmer* to reduce words to their base forms.
2. **Text Transformation:** Converting processed textual tokens into high-dimensional numerical matrices using **TF-IDF (Term Frequency-Inverse Document Frequency)** weighting.
3. **Multi-Kernel Benchmarking:** Conducting a comparative evaluation across 4 distinct SVM kernels: **Linear, RBF, Polynomial, and Sigmoid**.
4. **Data Ratio Splitting:** Training and testing the architectures across 3 multi-ratio data splits (**60:40, 70:30, and 80:20**) to validate model stability and generalization power.

## 📊 4. Experimental Results & Performance Analysis
Based on extensive benchmarking on the test data, the **SVM model configured with a Linear Kernel on an 80:20 data split** achieved the most optimal performance. The high-dimensional TF-IDF feature space proved highly effective for linear separation.

### Detailed Evaluation Matrix (80:20 Split - Linear Kernel):
* **Overall Accuracy:** **98.56%** (Indicating an exceptionally minimal classification error rate).
* **Valid News Class Performance:** Precision: `98.52%` | Recall: `99.25%` | F1-Score: `98.88%` (Support: 531 test data points).
* **Fake/Hoax News Class Performance:** Precision: `98.63%` | Recall: `97.30%` | F1-Score: `97.96%` (Support: 300 test data points).
* **Confusion Matrix Diagnosis:** Out of 831 total test documents, the model produced only **4 False Positives** and **8 False Negatives**, proving that the model is highly balanced, fair, and objective across both classes without any skewed bias.

## 🧠 5. Model Interpretability (Explainable AI via LIME)
The integration of LIME resolves the interpretability challenge of the SVM classifier by providing visual explanations on two distinct levels:

* **Global Interpretation:** The model successfully identified key distinguishing words. Strong indicators pushing a document towards the **Hoax** class are heavily dominated by informal, social-media-driven terms such as `telusur`, `unggah`, `youtube`, and `tiktok`. Conversely, indicators for **Valid** news are dominated by institutional, factual, and data-driven language like `persen`, `rabu`, `politik`, and `hadir`.
* **Local Interpretation:** On an individual test document level, LIME demonstrates that the model assigns predictions with absolute confidence (probabilities up to 1.00) when these localized clusters of key indicator words are present in the text.
