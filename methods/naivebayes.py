from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Contoh data latih
texts = ["Mata saya mulai terasa berat.", "Saya merasa sangat segar dan siap mengemudi.", "Saya harus berkonsentrasi saat mengemudi."]
labels = [1, 0, 1]  # 1 untuk kantuk, 0 untuk tidak kantuk

# Membuat vektor fitur dari teks
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Inisialisasi dan latih model Naive Bayes
model = MultinomialNB()
model.fit(X, labels)

# Contoh prediksi
new_texts = ["Saya merasa pusing.", "Saya bangun setelah tidur panjang dan segar."]
new_X = vectorizer.transform(new_texts)
predictions = model.predict(new_X)

print("Predictions:", predictions)
