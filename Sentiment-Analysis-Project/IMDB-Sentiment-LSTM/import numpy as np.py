import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

top_words = 10000
max_review_length = 300

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
X_train = pad_sequences(X_train, maxlen=max_review_length)
X_test = pad_sequences(X_test, maxlen=max_review_length)

model = Sequential()
model.add(Embedding(top_words, 128, input_length=max_review_length))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=6,
    batch_size=64,
    callbacks=[early_stop]
)

loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100))

model.save('sentiment_model.keras')

from tensorflow.keras.models import load_model
model = load_model('sentiment_model.keras')

word_index = imdb.get_word_index()
index_word = {v+3: k for k, v in word_index.items()}
index_word[0] = "<PAD>"
index_word[1] = "<START>"
index_word[2] = "<UNK>"
index_word[3] = "<UNUSED>"

def encode_review(text):
    words = text.lower().split()
    encoded = [1]  # <START>
    for word in words:
        if word in word_index and word_index[word] < top_words:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)  # <UNK>
    return pad_sequences([encoded], maxlen=max_review_length)

def predict_sentiment(text):
    encoded = encode_review(text)
    prediction = model.predict(encoded, verbose=0)[0][0]
    if prediction < 0.4:
        return "Negative"
    elif prediction > 0.6:
        return "Positive"
    else:
        return "Neutral"

while True:
    user_review = input("\nEnter your review (type 'exit' to stop): ")
    if user_review.lower() in ["exit", "quit"]:
        print("Session Ended.")
        break
    print("Predicted Sentiment:", predict_sentiment(user_review))