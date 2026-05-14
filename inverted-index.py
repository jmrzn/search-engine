import os
import json
from nltk import download

download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from collections import defaultdict

index = defaultdict(list)
unique_tokens = set()
REPORT = "report.txt"

stemmer = PorterStemmer()

def tokenize_text(text):
    result = []
    tokens = word_tokenize(text)
    
    for token in tokens:
        token = token.lower()
        if not token.isalpha():
            continue
        token = stemmer.stem(token)
        result.append(token)
    return result

def parse_content(content):
    if content.strip().startswith("<?xml"):
        return BeautifulSoup(content, features="xml")
    return BeautifulSoup(content, "html.parser")

def process_directory(root_path):
    doc_id_counter = 0
    # Iterate through domains in directory/root_path
    for domain in os.listdir(root_path):
        folder_path = os.path.join(root_path, domain)
        if not os.path.isdir(folder_path):  # add this
            continue
        
        # Iterate through each page/file in domain
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    content = data.get("content", "")
                    soup = parse_content(content)
                    clean_text = soup.get_text()
                    tokens = tokenize_text(clean_text)
                    add_to_index(doc_id_counter, tokens)
                    doc_id_counter += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return doc_id_counter

def add_to_index(doc_id, tokens):
    # Calculate term frequency
    term_freqs = defaultdict(int)
    for token in tokens:
        term_freqs[token] += 1
        unique_tokens.add(token)

    # A posting for docID and the term frequency
    for token, count in term_freqs.items():
        index[token].append({'docID': doc_id, 'term_freqs': count})

def save_index(output_file):
    with open(output_file, 'w') as f:
        json.dump(index, f)
    return os.path.getsize(output_file) / 1024  # Size in KB

def generate_report():
    doc_id_counter = process_directory('test_developer')
    size_kb = save_index('inverted_index.json')

    with open(REPORT, "w") as f:

        print(f"Documents Indexed: {doc_id_counter}", file=f)
        print(f"Unique Tokens: {len(unique_tokens)}", file=f)
        print(f"Index Size: {size_kb:.2f} KB", file=f)

if __name__ == "__main__":
    generate_report()