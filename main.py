# main.py
import pickle
import nltk
from nltk.corpus import stopwords
import string
from transformers import pipeline
import os
from tqdm import tqdm

# Check and download only if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set up
translator = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# main.py
project_root = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_root, "data", "newsgroups_data.pkl")

# Load from file
try:
    with open(data_path, "rb") as f:
        newsgroups = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}")
    print("Please run download_data.py first.")
    exit()
    
# Function to preprocess and summarize a document
def preprocess_and_summarize(document_text):
    # Preprocess the text
    processed_text = document_text.lower().translate(translator)

    # Estimate token count (roughly 1 word = 1.3 tokens on average for English)
    word_count = len(processed_text.split())
    token_estimate = int(word_count * 1.3)

    # Set max_length dynamically
    max_len = max(40, int(token_estimate * 0.5))  # 40 is a reasonable floor
    max_len = min(max_len, 150)  # cap it for performance

    # Perform summarization
    output = summarizer(
        processed_text,
        max_length=max_len,
        min_length=30,
        do_sample=False
    )

    return output[0]['summary_text']

# Example: Process the first N documents
num_docs_to_process = min(3, len(newsgroups.data))  # Or a larger number
print(f"Processing {num_docs_to_process} documents...\n")

summaries = []

if num_docs_to_process > 0:
    with tqdm(total=num_docs_to_process, desc="Summarizing Documents", ncols=80) as pbar:
        for i in range(num_docs_to_process):
            summary = preprocess_and_summarize(newsgroups.data[i])
            summaries.append((i, summary))
            pbar.update(1)
    
    # Use tqdm.write() instead of print() to avoid interfering with the progress bar
    for i, summary in summaries:
        tqdm.write(f"\n--- Document {i} ---")
        tqdm.write("Summary:")
        tqdm.write(summary)
else:
    print("No documents found in the dataset.")