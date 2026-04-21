import pandas as pd

def load_data(path):
    df = pd.read_parquet(path, columns = ["Consumer complaint narrative", "Product"])
    return df

def clean_data(df):
    df = df[df["Consumer complaint narrative"].notna()]
    return df

def map_labels(df):
    mapping = {
        "Credit reporting": "Credit reporting, repair, or other",
        "Credit reporting or other personal consumer reports": "Credit reporting, repair, or other",
        "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting, repair, or other",

        "Credit card": "Credit card or prepaid card",
        "Prepaid card": "Credit card or prepaid card",

        "Money transfers": "Money transfer, virtual currency, or money service",
        "Virtual currency": "Money transfer, virtual currency, or money service",

        "Payday loan": "Loan",
        "Payday loan, title loan, or personal loan": "Loan",
        "Payday loan, title loan, personal loan, or advance loan": "Loan",
        "Consumer Loan": "Loan",
        "Vehicle loan or lease": "Loan",
        "Student loan": "Loan"
    }

    df['Product'] = df['Product'].replace(mapping)
    return df

    