import pickle
import nltk
from nltk.tokenize import word_tokenize

# Check and download required NLTK data only if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Load from file
with open("newsgroups_data.pkl", "rb") as f:
    newsgroups = pickle.load(f)


first_doc = newsgroups.data[0].lower()
first_doc_tokenized = word_tokenize(first_doc)
print(first_doc_tokenized)