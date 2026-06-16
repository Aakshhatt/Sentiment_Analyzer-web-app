import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data
nltk.download('stopwords')

# ✅ Sample Training Data (positive and negative sentences)
texts = [
    "I love this movie", "This is amazing", "Absolutely wonderful experience",
    "Best thing ever", "I really enjoyed this", "Fantastic and brilliant",
    "Great product highly recommend", "So happy with this purchase",
    "Excellent quality and service", "Very good and satisfying",
    "I hate this movie", "This is terrible", "Worst experience ever",
    "Absolutely horrible", "I really disliked this", "Very disappointing",
    "Bad product do not buy", "So unhappy with this purchase",
    "Poor quality and service", "Very bad and unsatisfying"
]

# ✅ Labels — 1 means Positive, 0 means Negative
labels = [1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0]

# ✅ Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# ✅ Train the AI model
model = LogisticRegression()
model.fit(X, labels)

# ✅ Save the model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully!")