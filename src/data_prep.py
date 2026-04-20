import pandas as pd
import pyarrow

def run_data_pipeline():

    #step1:import raw data csv
    raw_data = pd.read_csv("data/raw/complaints.csv")

    #step2:convert csv to parquet because it will save some space
    raw_data.to_parquet("data/raw/complaints.parquet")

    #step3: sample 1000000 rows from the parquet file to form our core data
    data_sample = raw_data.sample(n=1000000, random_state=42)
    
    #step4: save the sampled data as parquet inside data folder
    data_sample.to_parquet("data/processed/data.parquet")

if __name__ == "__main__":
    run_data_pipeline()
