from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pickle


def traintest_split(df):
    X = df["Consumer complaint narrative"]
    y = df["Product"]
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    return X_train,X_test,y_train,y_test

def vectorize_text(X_train,X_test, vectorizer_path):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    pickle.dump(vectorizer,open(vectorizer_path,"wb"))
    return X_train_tfidf, X_test_tfidf, vectorizer

def model(X_train,y_train,model_path):
    model = LinearSVC(class_weight="balanced", max_iter= 5000, dual = "auto")
    model.fit(X_train, y_train)
    pickle.dump(model, open(model_path, "wb"))
    return model

def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred)) 