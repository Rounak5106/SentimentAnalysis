import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_text
import joblib

data = pd.read_csv('data/twitter_sentiment.csv')

data['clean_text'] = data['text'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_text'])
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save vectorizer and model
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'nb_model.pkl')

# Example prediction
new_text = ["I love this product, it's amazing!"]
new_text_clean = [preprocess_text(text) for text in new_text]
new_text_vector = vectorizer.transform(new_text_clean)
prediction = model.predict(new_text_vector)
print("Predicted Sentiment:", prediction)