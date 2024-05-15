from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

# Load your data
# For simplicity, let's assume X and y are your features and labels

data = []
words = []

with open('Data/Train data.csv') as f:
  reader = csv.reader(f)
  next(reader)
  for row in reader:

    # Tokenisation of words for embedding creation
    t = int(row[0])
    title = row[1].split()
    title = [t.lower() for t in title]
    description = row[2].split()
    description = [d.lower() for d in description]
    try:
      words.extend(t.lower() for t in title)
    except:
      pass

    data.append([t,title,description]) # List containing the index title and description


X_train, X_test, y_train, y_test = train_test_split(title, t, test_size=0.2, random_state=42)

# Transform text data to feature vectors
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Define baseline models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "NaiveBayes": MultinomialNB()
}

# Evaluate each model
for name, model in models.items():
    model.fit(X_train_vect, y_train)
    y_pred = model.predict(X_test_vect)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")
