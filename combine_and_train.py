import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Define column names for the LIAR dataset
column_names = [
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "state", "party", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
]

# Load LIAR dataset files
df_train = pd.read_csv("train.tsv", sep="\t", header=None, names=column_names)
df_test = pd.read_csv("test.tsv", sep="\t", header=None, names=column_names)
df_valid = pd.read_csv("valid.tsv", sep="\t", header=None, names=column_names)

# Load your custom CSV
df_custom = pd.read_csv("custom_news.csv")  # Should have 'statement' and 'label' columns

# Combine all datasets
df_all = pd.concat([
    df_train[['statement', 'label']],
    df_test[['statement', 'label']],
    df_valid[['statement', 'label']],
    df_custom[['statement', 'label']]
], ignore_index=True)

# Optional: check label distribution
print(df_all['label'].value_counts())

# Split combined data
X_train, X_test, y_train, y_test = train_test_split(
    df_all['statement'], df_all['label'], test_size=0.2, random_state=42
)

# Vectorizer + Model pipeline
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score * 100, 2)}%")
print(confusion_matrix(y_test, y_pred))

# Save model and vectorizer
with open("final_model.sav", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Model and vectorizer saved successfully!")
