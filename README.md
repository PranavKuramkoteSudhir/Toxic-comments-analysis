# Toxic Comments Analysis

This project implements a machine learning pipeline to detect and classify toxic comments across multiple categories. It uses the Jigsaw Toxic Comment Classification dataset to identify different types of toxicity like threats, obscenity, insults, and identity-based hate in text comments.

## Problem Statement
Online toxicity is a significant challenge in digital communication. This project aims to automatically identify and classify toxic comments to help maintain healthy online discussions. The model classifies comments into six categories:
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

## Technical Architecture
The project follows a modular architecture with the following components:

project/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_training.py
│   └── pipeline/
│       └── prediction_pipeline.py
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
├── templates/
│   ├── css/
│   │   └── style.css
│   ├── home.html
│   └── index.html
└── tests/
    ├── __init__.py
    └── test_app.py

1. **Data Ingestion (`data_ingestion.py`)**
   - Handles data loading and train-test splitting
   - Creates necessary artifacts directory
   - Implements stratified splitting to handle class imbalance

2. **Data Transformation (`data_transformation.py`)**
   - Text preprocessing and cleaning
   - Feature engineering including:
     - TF-IDF vectorization
     - Text length and word count features
     - Special character and punctuation analysis
   - Implements sklearn Pipeline for reproducible transformations

3. **Model Training (Upcoming)**
   - Multi-label classification
   - Model evaluation and selection
   - Hyperparameter tuning

## Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Regular Expressions (Re)

## Getting Started

### Prerequisites
- Python 3.x
- Virtual Environment (recommended)
