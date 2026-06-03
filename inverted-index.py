import os
import json
from nltk import download

download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from collections import defaultdict
from simhash import Simhash

index = defaultdict(list)
unique_tokens = set()
REPORT = "report.txt"
SIMHASH_THRESHOLD = 3

stemmer = PorterStemmer()

def is_near_duplicate(fp, seen_fingerprints, threshold=SIMHASH_THRESHOLD):
    for seen in seen_fingerprints:
        if fp.distance(seen) <= threshold:
            return True
    return False

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

def get_features(tokens):
    return [' '.join(tokens[i:i+2]) for i in range(len(tokens) - 1)]

def process_directory(root_path):
    doc_id_counter = 0
    duplicates_skipped = 0
    seen_fingerprints = []
    doc_id_to_url = {}

    for domain in os.listdir(root_path):
        folder_path = os.path.join(root_path, domain)
        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    content = data.get("content", "")
                    url = data.get("url", "")
                    soup = parse_content(content)
                    clean_text = soup.get_text()
                    tokens = tokenize_text(clean_text)

                    if len(tokens) < 50:
                        duplicates_skipped += 1
                        continue

                    fp = Simhash(get_features(tokens))
                    if is_near_duplicate(fp, seen_fingerprints):
                        duplicates_skipped += 1
                        print(f"Skipped: {url}")
                        continue
                    seen_fingerprints.append(fp)

                    doc_id_to_url[doc_id_counter] = url
                    add_to_index(doc_id_counter, tokens)
                    print(f"Adding doc {doc_id_counter} to index")
                    doc_id_counter += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    print(f"Duplicates skipped: {duplicates_skipped}")
    return doc_id_counter, doc_id_to_url

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

def save_url_map(doc_id_to_url, output_file="url_map.json"):
    with open(output_file, 'w') as f:
        json.dump(doc_id_to_url, f)

def generate_report():
    doc_id_counter, doc_id_to_url = process_directory('ANALYST')
    size_kb = save_index('inverted_index.json')
    save_url_map(doc_id_to_url)

    with open(REPORT, "w") as f:

        print(f"Documents Indexed: {doc_id_counter}", file=f)
        print(f"Unique Tokens: {len(unique_tokens)}", file=f)
        print(f"Index Size: {size_kb:.2f} KB", file=f)

if __name__ == "__main__":
    generate_report()
