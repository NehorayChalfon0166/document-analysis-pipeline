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

#set up
translator = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load from file
with open("newsgroups_data.pkl", "rb") as f:
    newsgroups = pickle.load(f)

#readies the pur etext before tokenazation
first_doc = newsgroups.data[0].lower()
first_doc = first_doc.translate(translator)

#tokenize text
first_doc_tokenized = word_tokenize(first_doc)
doc_tokenized_no_stop_words = [token for token in first_doc_tokenized if token not in stop_words] #remove stop words

# Create a summarization pipeline an LLM
output = summarizer(first_doc)
print(output)