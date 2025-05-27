import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load your dataset (replace 'your_dataset.csv' with your actual file)
data = pd.read_csv("C:\Users\thami\Downloads\Reddit-20250525T101728Z-1-001\Reddit")

# Example columns: 'text' for news content, 'label' for fake/real
X = data['Americans Aren't Sure If Flight 370 Vanished Thanks to Aliens, Terrorists, or Hide-and-Seek']
y = data['1']

# Split dataset into training and testing sets (optional but recommended)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline combining TF-IDF vectorizer and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', LogisticRegression(solver='lbfgs', max_iter=1000))
])

# Train the model
pipeline.fit(X_train, y_train)

# (Optional) Check accuracy
print(f"Test Accuracy: {pipeline.score(X_test, y_test):.4f}")

# Save the trained model pipeline
joblib.dump(pipeline, 'fake_news_model.pkl')
