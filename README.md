# SentimentAnalysis
# Sentiment Analysis Tool

## Overview
Developed during my internship at **Micro IT**, this Python-based Sentiment Analysis tool leverages AI/ML and NLP to classify text as positive or negative, demonstrated using the **Sentiment140** Twitter dataset (~1.6 million tweets). The tool employs two models—**Multinomial Naive Bayes** (traditional ML, ~75–85% accuracy) and **DistilBERT** (transformer-based deep learning, ~85–95% accuracy)—to analyze tweet sentiments, showcasing a comparison of classical and modern NLP approaches. Key features include:

- **Text Preprocessing**: Cleans tweets by removing URLs, hashtags, and stopwords using NLTK.
- **Model Comparison**: Evaluates Naive Bayes as a fast baseline against BERT’s contextual analysis.
- **Visualizations**: Generates a pie chart (`sentiment_distribution.png`) for sentiment distribution and a confusion matrix (`confusion_matrix.png`) for model performance.
- **Extensible Design**: Applicable to other text data (e.g., reviews, comments) with minimal modification.

The tool includes comprehensive documentation, unit tests, and a LaTeX presentation, highlighting skills in NLP, machine learning, and data visualization.

## Objectives
- Build a flexible sentiment analysis system for text classification.
- Apply NLP techniques for effective text preprocessing.
- Compare traditional and deep learning models for performance.
- Visualize results for clear, actionable insights.

## Repository Structure
