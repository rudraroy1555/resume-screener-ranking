# NLP Resume Screening & Ranking Engine

## Overview
This project is an automated, two-stage Natural Language Processing (NLP) pipeline designed to screen and rank candidate resumes against a specific job description. It eliminates manual review by utilizing machine learning classification and mathematical text similarity.

## Pipeline Architecture

### Stage 1: The ML Gatekeeper (Supervised Learning)
Before ranking, resumes are passed through a Logistic Regression classifier trained on a historical HuggingFace dataset of labeled resumes. 
* **Function:** Predicts the broad industry category of the incoming resume (e.g., Data Science, HR, Web Design).
* **Action:** Instantly filters out candidates whose predicted category fundamentally mismatches the target role, saving computational overhead.

### Stage 2: The Scoring Engine (Unsupervised Learning)
Candidates who survive the Stage 1 filter are mathematically ranked against the core job description.
* **Preprocessing:** Raw text is standardized using NLTK (lowercasing, regex punctuation removal, stop-word filtering, and lemmatization).
* **Vectorization:** Cleaned text is transformed into a mathematical matrix using scikit-learn's TF-IDF (Term Frequency-Inverse Document Frequency) algorithm, which penalizes common words and emphasizes rare, highly specific keywords.
* **Scoring:** The system calculates the Cosine Similarity between the Job Description vector and the Candidate vectors. This measures the geometric angle between the texts, ensuring candidates are ranked purely on conceptual alignment rather than resume length.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas
* **Machine Learning:** scikit-learn (Logistic Regression, TfidfVectorizer, Cosine Similarity)
* **NLP Processing:** NLTK (WordNet, Punkt)

## How to Run
This project is hosted in a Google Colab notebook for frictionless execution. 
1. Click the "Open in Colab" badge at the top of the `.ipynb` file in this repository.
2. Select `Runtime > Run All` to execute the pipeline.
