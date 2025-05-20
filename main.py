import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from transformers import pipeline

# Check and download only if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# Load from file
with open("newsgroups_data.pkl", "rb") as f:
    newsgroups = pickle.load(f)

stop_words = set(stopwords.words('english'))
first_doc = newsgroups.data[0].lower()
first_doc_tokenized = word_tokenize(first_doc)

doc_tokenized_no_stop_words = [token for token in first_doc_tokenized if token not in stop_words]
cleaned_doc = [word for word in doc_tokenized_no_stop_words if not all(char in string.punctuation for char in word)]
cleaned_text = " ".join(cleaned_doc)
# Create a summarization pipeline using the 'facebook/bart-large-cnn' model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
output = summarizer(cleaned_text)
print(output)