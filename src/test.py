import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
vectorizer_path_name = os.path.join(BASE_DIR,"artifacts/vectorizer.pkl")
model_path_name = os.path.join(BASE_DIR,"artifacts/model.pkl")

model = pickle.load(open(model_path_name, "rb"))
vectorizer = pickle.load(open(vectorizer_path_name, "rb"))

text = ""

text_tfidf = vectorizer.transform([text])
prediction = model.predict(text_tfidf)

print(prediction)