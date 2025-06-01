# src/sentiment_analysis_bert.py
from transformers import pipeline
from preprocess import preprocess_text

# Load BERT model
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Example prediction
sample_text = ["This product is fantastic!", "Terrible service, very disappointed."]
clean_text = [preprocess_text(text) for text in sample_text]
results = classifier(clean_text)
for text, result in zip(sample_text, results):
    print(f"Text: {text}\nSentiment: {result['label']}, Score: {result['score']:.4f}\n")