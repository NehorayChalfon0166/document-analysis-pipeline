import pickle

# Load from file
with open("newsgroups_data.pkl", "rb") as f:
    newsgroups = pickle.load(f)

# Use the data
print(f"Number of documents: {len(newsgroups.data)}")
print(f"Target names (categories): {newsgroups.target_names}")
print(f"\nFirst document:\n{newsgroups.data[0][:200]}...")
print(f"\nFirst document's category: {newsgroups.target_names[newsgroups.target[0]]}")
