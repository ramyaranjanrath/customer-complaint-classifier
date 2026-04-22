import os
import pickle

#get to the folder for artifacts
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

#load model and vectorizer
model = pickle.load(open(os.path.join(ARTIFACT_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(ARTIFACT_DIR, "vectorizer.pkl"), "rb"))

#write prediction function
def prediction(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction[0]