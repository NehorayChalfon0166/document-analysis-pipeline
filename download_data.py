# download_data.py
from sklearn.datasets import fetch_20newsgroups
import pickle
import os

project_root = os.path.dirname(os.path.abspath(__file__))  # Gets directory of current script
save_folder = os.path.join(project_root, "data")
os.makedirs(save_folder, exist_ok=True)

# File path to save
save_path = os.path.join(save_folder, "newsgroups_data.pkl")

# Check if the dataset already exists
if os.path.exists(save_path):
    print(f"Dataset already found at {save_path}. Skipping download.")
else:
    print("Downloading dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    print("Saving dataset locally...")
    with open(save_path, "wb") as f:
        pickle.dump(newsgroups, f)

    print(f"Dataset saved locally at {save_path}")
