import os
import pandas as pd
import pyarrow
import requests


def download_file(url, save_path):
    #if the file already exists there we won't download it again.
    if os.path.exists(save_path):
        print("file already exists")
        return
    #if the file doesn't exist, we will download the file
    print("Downloading the file....")
    #get the file from the internet and when stream = True, the file won't be downloaded at once
    response = requests.get(url, stream= True)
    #since stream = True, we will now download it in chunks of size 8192
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Download Complete")


def create_dataset(path,save_path, sample=False, n=2000000):
    #step1:import raw data csv
    raw_data = pd.read_csv(path)
    if sample:
        raw_data = raw_data.sample(n=n, random_state= 42)
    #step2:convert csv to parquet because it will save some space
    raw_data.to_parquet(save_path)
    print("Dataset created")
