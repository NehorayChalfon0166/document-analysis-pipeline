import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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
print(doc_tokenized_no_stop_words)