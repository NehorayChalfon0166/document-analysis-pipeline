# main.py
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from transformers import pipeline
import os

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
    processed_text = document_text.lower().translate(translator)  # Use original text for summarizer for now
    # tokenized_for_other_tasks = [token for token in word_tokenize(processed_text) if token not in stop_words]

    # BART performs better with the full, less preprocessed text 
    # Becasuse they handle tokenization internally, and stop words can provide context.
    output = summarizer(document_text, max_length=150, min_length=30, do_sample=False)  # BART takes raw text
    print("Summary:")
    print(output[0]['summary_text'])  # Access the summary text
    # return output[0]['summary_text'], tokenized_for_other_tasks  # if you need them

# Example: Process the first 3 documents
for i in range(min(3, len(newsgroups.data))):
    print(f"--- Processing Document {i} ---")
    preprocess_and_summarize(newsgroups.data[i])

# for specific document:
# preprocess_and_summarize(newsgroups.data[5])