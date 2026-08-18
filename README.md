# NLP Resume Screening & Ranking Engine

## Overview
This project is an automated, two-step Natural Language Processing pipeline for filtering and ranking candidate resumes against a target job description. The system screens resumes using machine learning classification before ranking them via mathematical text similarity.

---
## Technical Workflow

### 1. Stage 1: The ML Gatekeeper
A Logistic Regression classifier trained on a historical HuggingFace resume dataset predicts the general category of the incoming resume. This acts as a filter, instantly dropping resumes that do not match the target job category.

### 2. Stage 2: The Scoring Engine
Surviving resumes are mathematically scored against the core job description. Cleaned text is converted into a matrix of word vectors using scikit-learn's `TfidfVectorizer`. We then apply Cosine Similarity to calculate the geometric angle between the vectors, ranking the conceptual alignment regardless of resume length.

---
## Text Cleaning & Preprocessing
Prior to scoring, raw text is standardized using NLTK:
* Lowercase conversion
* Regex pattern matching for punctuation removal
* Stop-word removal
* Word lemmatization (reduction to root forms)

---
## Tech Stack
* **Language:** Python
* **Data Engineering:** Pandas
* **Machine Learning:** scikit-learn
* **NLP Processing:** NLTK

---
## Usage
This pipeline is hosted in a Google Colab notebook for frictionless execution.
1. Click the **Open in Colab** badge at the top of the `.ipynb` file in this repository.
2. Select `Runtime > Run All` to execute the pipeline.
