from data_ingestion import create_dataset
from data_processing import load_data, clean_data, map_labels
from model import traintest_split, vectorize_text, model, evaluate_model
import os 

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR,"artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Step 1: create dataset
#create_dataset("data/raw/complaints.csv", "data/processed/data.parquet", sample=True)

# Step 2: load processed data
df = load_data("data/processed/data.parquet")

# Step 3: clean + map
df = clean_data(df)
df = map_labels(df)

# Step 4: split
X_train, X_test, y_train, y_test = traintest_split(df)

# Step 5: vectorize
vectorizer_path = os.path.join(ARTIFACT_DIR, "vectorizer.pkl")
X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test,vectorizer_path)

# Step 6: train model
model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
clf = model(X_train_tfidf, y_train, model_path)

# Step 7: evaluate
evaluate_model(clf, X_test_tfidf, y_test)