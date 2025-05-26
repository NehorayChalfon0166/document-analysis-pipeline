# Document Summarization Pipeline with 20 Newsgroups

This project demonstrates a pipeline for downloading the 20 Newsgroups dataset, preprocessing text documents, and generating summaries using the `facebook/bart-large-cnn` model from Hugging Face Transformers.

## Features

*   **Data Fetching:** Downloads the 20 Newsgroups dataset using `scikit-learn` and saves it locally to avoid repeated downloads.
*   **Text Preprocessing:**
    *   Converts text to lowercase.
    *   Removes punctuation.
    *   (NLTK tokenization and stop word lists are available but not directly used for the BART summarizer input in the current version).
*   **Document Summarization:**
    *   Utilizes the `facebook/bart-large-cnn` model via the Hugging Face `pipeline` for abstractive summarization.
    *   Dynamically adjusts the `max_length` parameter for the summarizer based on an estimate of the input document's token count.
    *   Processes a configurable number of documents (defaults to the first 3).
*   **Progress Tracking:** Uses `tqdm` to display a progress bar during the summarization process.
*   **Automatic Resource Download:**
    *   NLTK automatically downloads necessary resources (`punkt` tokenizer and `stopwords` corpus) if not found.
    *   The Transformers library downloads the `facebook/bart-large-cnn` model on its first use.

## Tech Stack / Dependencies

*   Python 3.x
*   **scikit-learn:** For fetching the 20 Newsgroups dataset.
*   **NLTK:** For text preprocessing utilities (though full tokenization/stopword removal isn't fed to BART).
*   **Hugging Face Transformers:** For the summarization pipeline and model.
*   **PyTorch (torch):** As a backend for Hugging Face Transformers.
*   **tqdm:** For creating progress bars.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```
    scikit-learn
    nltk
    transformers
    torch
    tqdm
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first time you run `main.py`, NLTK may download `punkt` and `stopwords`. Similarly, Hugging Face Transformers will download the `facebook/bart-large-cnn` model if it's not already cached.*

## Usage

1.  **Download the Dataset:**
    Run the `download_data.py` script. This will fetch the 20 Newsgroups dataset and save it to a `data/` subdirectory within your project. This step only needs to be performed once.
    ```bash
    python download_data.py
    ```
    You should see output indicating whether the dataset was downloaded or if it was already found.

2.  **Run the Summarization Pipeline:**
    Execute the `main.py` script. This will load the dataset, preprocess the specified number of documents, and generate summaries.
    ```bash
    python main.py
    ```
    The script will display a progress bar and then print the summaries for each processed document to the console.

    **Customization:**
    You can change the number of documents to process by modifying the `num_docs_to_process` variable in `main.py`:
    ```python
    # main.py
    # ...
    num_docs_to_process = min(5, len(newsgroups.data)) # Process first 5, for example
    # ...
    ```

## Project Structure
document-analysis-pipeline/
├── data/ # (Created by download_data.py) Stores newsgroups_data.pkl
├── download_data.py # Script to download the 20 Newsgroups dataset
├── main.py # Main script for preprocessing and summarization
├── requirements.txt # Python package dependencies
└── README.md # This file


## Potential Future Work

*   Allow processing a specific document by index via a command-line argument.
*   Save generated summaries to a file (e.g., CSV, JSON, or text files).
*   Explore other summarization models from Hugging Face Hub.
*   Implement more advanced text preprocessing or feature extraction steps if expanding to other NLP tasks.
*   Add evaluation metrics for summarization (e.g., ROUGE scores, if reference summaries were available or generated).