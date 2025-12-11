import os
import zipfile
import requests
import subprocess
import pandas as pd


KAGGLE_DATASET = 'amirshnll/persian-sms-dataset'
INSTAGRAM_DATASET1 = "https://raw.githubusercontent.com/davardoust/PHICAD/refs/heads/main/PHICAD-part1.csv"
INSTAGRAM_DATASET2 = "https://raw.githubusercontent.com/davardoust/PHICAD/refs/heads/main/PHICAD-part2.csv"
SPAM_EMAILS_DATASET = "https://raw.githubusercontent.com/Melanee-Melanee/HamSpam-EMAIL/refs/heads/main/src/emails.csv"
DATASET_FOLDER = "Data/"


def _download_kaggle_dataset(
                        kaggle_username: str,
                        kaggle_key: str,
                        dataset_name: str = KAGGLE_DATASET,
                    ):
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    command = [
        'kaggle', 
        'datasets', 
        'download',
        '-d', dataset_name,
        '-p', DATASET_FOLDER,
        '--unzip'
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    


def _download_csv_dataset(
                    url: str, 
                    filename: str, 
                    columns: list[str]
                ) -> pd.DataFrame:
    if not os.path.exists(filename):
        content = requests.get(url, stream=True)
        with open(filename, mode='wb') as f:
            for data in content.iter_content(chunk_size=8192):
                if data:
                    f.write(data)
    return pd.read_csv(filename, sep='\t', names=columns, header=None, encoding='utf-8')


def download_and_load_data(
            kaggle_username: str,
            kaggle_key: str
        ):
    _download_kaggle_dataset(kaggle_username, kaggle_key)
    first_df = _download_csv_dataset(INSTAGRAM_DATASET1, f"{DATASET_FOLDER}instagram_dataset1.csv", ["comment_normalized", "thate", "tspam", "tobscene", "tclass"])
    second_df = _download_csv_dataset(INSTAGRAM_DATASET2, f"{DATASET_FOLDER}instagram_dataset2.csv", ["comment_normalized", "thate", "tspam", "tobscene", "tclass"])
    third_df = _download_csv_dataset(SPAM_EMAILS_DATASET, f"{DATASET_FOLDER}spam_emails.csv", None)
    spam_sms_df = pd.read_csv("Data/sms_data.txt", encoding='utf-16', header=None, names=["comment_normalized"])
    spam_sms_df = spam_sms_df.replace(r'\t+', '', regex=True)

    return first_df, second_df, third_df, spam_sms_df