from sklearn.datasets import fetch_20newsgroups
import pickle
from tqdm import tqdm
import os

# current project folder
save_folder = r"C:\Users\User\projects\document-analysis-pipeline"
os.makedirs(save_folder, exist_ok=True)

# File path to save
save_path = os.path.join(save_folder, "newsgroups_data.pkl")

# Download dataset
print("Downloading dataset...")
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

print("Saving dataset locally...")
with open(save_path, "wb") as f:
    pickle.dump(newsgroups, f)

print(f"Dataset saved locally at {save_path}")
