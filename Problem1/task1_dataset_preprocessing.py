"""
CSL 7640 - Assignment 2 (Problem 1)

Task: Dataset Collection + Preprocessing + Statistics + WordCloud

This script performs the following steps:
1. Collects textual data from multiple IIT Jodhpur sources
2. Extracts only clean English text (removes boilerplate + non-English)
3. Preprocesses text (lowercase, remove punctuation, clean spaces)
4. Tokenizes the text
5. Computes dataset statistics
6. Generates a WordCloud

"""

# IMPORTS 

import requests                      # for fetching web pages
from bs4 import BeautifulSoup       # for parsing HTML
import re                           # for text cleaning (regex)
import os                           # for file handling
import time                         # to delay requests 
from collections import Counter     # for word frequency
from wordcloud import WordCloud     # for visualization
import matplotlib.pyplot as plt


#  CONFIG 

# Directory to store raw text files
RAW_DIR = "raw_corpus"
os.makedirs(RAW_DIR, exist_ok=True)

# Polite delay between requests (avoids overloading server)
DELAY = 1.5

# User-agent header to mimic browser (prevents blocking)
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# IIT Jodhpur sources
SOURCES = {
    # Academic programs 
    "programs": "https://www.iitj.ac.in/office-of-academics/en/list-of-academic-programs",

    # Academic regulations
    "regulations": "https://www.iitj.ac.in/office-of-academics/en/academic-regulations",

    # Academic programs overview
    "academics_overview": "https://www.iitj.ac.in/office-of-academics/en/academic-programs",

    # Research areas (CSE department)
    "research": "https://iitj.ac.in/computer-science-engineering/en/research-area-labs",

    # Faculty members 
    "faculty": "https://www.iitj.ac.in/main/en/faculty-members"
}

#  STEP 1: DATA COLLECTION 

def fetch_and_clean(url):
    """
    Fetch HTML from a URL and extract clean English text.

    Steps:
    1. Remove script/style tags (boilerplate)
    2. Extract visible text
    3. Filter out non-English lines using ASCII heuristic
    """

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted HTML elements (navigation, scripts, etc.)
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract raw text
        text = soup.get_text(separator="\n")

        clean_lines = []

        for line in text.split("\n"):
            line = line.strip()

            # Skip very short lines (likely noise)
            if len(line) < 30:
                continue

            # Keep only English-like lines (ASCII heuristic)
            alpha_chars = [c for c in line if c.isalpha()]
            if not alpha_chars:
                continue

            ascii_ratio = sum(1 for c in alpha_chars if ord(c) < 128) / len(alpha_chars)
            if ascii_ratio < 0.8:
                continue  # skip non-English lines

            # Remove URLs and extra spaces
            line = re.sub(r"http\S+", "", line)
            line = re.sub(r"\s+", " ", line)

            clean_lines.append(line)

        return "\n".join(clean_lines)

    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


def collect_data():
    """
    Collect data from all sources and save into separate text files.
    """
    print("\n=== DATA COLLECTION STARTED ===\n")

    for name, url in SOURCES.items():
        print(f"[Fetching] {name}")

        text = fetch_and_clean(url)

        file_path = os.path.join(RAW_DIR, f"{name}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved: {file_path} ({len(text)} chars)")

        time.sleep(DELAY)

    print("\n=== DATA COLLECTION COMPLETE ===\n")


#STEP 2: BUILD CORPUS 

def build_corpus():
    """
    Combine all text files into a single corpus.
    """
    all_text = []

    for file in os.listdir(RAW_DIR):
        if file.endswith(".txt"):
            with open(os.path.join(RAW_DIR, file), "r", encoding="utf-8") as f:
                all_text.append(f.read())

    corpus = "\n".join(all_text)

    with open("corpus.txt", "w", encoding="utf-8") as f:
        f.write(corpus)

    print("Corpus file created: corpus.txt")
    return corpus


# STEP 3: PREPROCESSING 

def preprocess(text):
    """
    Clean text by:
    1. Lowercasing
    2. Removing punctuation
    3. Removing extra spaces
    """

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text


def tokenize(text):
    """
    Convert text into list of words (tokens).
    """
    return text.split()


# STEP 4: STATISTICS 

def compute_statistics(tokens):
    """
    Compute dataset statistics:
    - total documents
    - total tokens
    - vocabulary size
    """

    total_docs = len(os.listdir(RAW_DIR))
    total_tokens = len(tokens)
    vocab = set(tokens)

    print("\n=== DATASET STATISTICS ===")
    print("Total documents:", total_docs)
    print("Total tokens:", total_tokens)
    print("Vocabulary size:", len(vocab))

    return Counter(tokens)


#  STEP 5: WORDCLOUD 

def generate_wordcloud(text):
    """
    Generate and display word cloud for most frequent words.
    """

    wc = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Word Cloud of IIT Jodhpur Corpus")

    import os
    save_path = os.path.join(os.getcwd(), "wordcloud.png")
    plt.savefig(save_path)

    print("Wordcloud saved at:", save_path)

    plt.show()

# MAIN EXECUTION

if __name__ == "__main__":

    # Step 1: Collect data
    collect_data()

    # Step 2: Build corpus
    raw_text = build_corpus()

    # Step 3: Preprocess
    clean_text = preprocess(raw_text)

    # Step 4: Tokenize
    tokens = tokenize(clean_text)

    # Step 5: Statistics
    freq = compute_statistics(tokens)

    # Step 6: WordCloud
    generate_wordcloud(clean_text)