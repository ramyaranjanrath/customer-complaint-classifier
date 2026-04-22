# Customer Complaint Classification (End-to-End ML Project)
******************************************************************************************************************************************
- Prediction dashboard link: [https://classifying-customer-complaint.streamlit.app/ 
](https://customer-complaint-classifier.streamlit.app/)

- This project builds and deploys a machine learning system that classifies customer complaint text into product categories using TF-IDF and LinearSVC.  
- The model is deployed as an interactive web app using Streamlit.

## Problem Statement

Customer complaints contain unstructured text. The goal is to automatically classify each complaint into a relevant product category (e.g., credit card, loan, bank account).

---
## Approach

The project follows an end-to-end ML pipeline:

1. Data ingestion and sampling  
2. Data cleaning and label mapping  
3. Text vectorization using TF-IDF  
4. Model training using LinearSVC  
5. Model evaluation using classification metrics  
6. Saving trained model and vectorizer  
7. Deployment using Streamlit  


---
## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn (TF-IDF, LinearSVC)  
- Streamlit  

---
## Project Structure

src/

data_ingestion.py

data_processing.py

model.py

inference.py

artifacts/

model.pkl

vectorizer.pkl

app.py

requirements.txt

README.md

---
## How to Run Locally

- Clone the repository:https://github.com/ramyaranjanrath/customer-complaint-classifier
- cd your-repo-name
- pip install -r requirements.txt
- Run the Streamlit app: streamlit run app.py

---
## Example

- **Input:** My bank charged me extra fees without explanation
- **Output:** Checking or savings account

---

## Key Learnings

- Built a complete ML pipeline from raw data to deployment  
- Understood text preprocessing and TF-IDF feature extraction  
- Worked with imbalanced datasets and model evaluation  
- Learned to structure code into reusable modules  
- Deployed an ML model as a web application  

---

## Future Improvements

- Improve model performance with advanced NLP techniques  
- Add FastAPI backend for scalable inference  
- Track experiments using MLflow  
- Improve UI and user experience  

---

## Author

Ramyaranjan Rath
