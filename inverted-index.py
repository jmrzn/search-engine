import os
import json
from bs4 import BeautifulSoup
from collections import defaultdict

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
    "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves",
}

index = defaultdict(list)
unique_tokens = set()
REPORT = "report.txt"

def tokenize_text(text):
    tokens = []
    current = []
 
    for char in text:
        try:
            if char.isalnum() and char.isascii():
                current.append(char.lower())
            elif char == "'" and current:
                # allow apostrophe in the middle of a word
                current.append(char)
            elif current:
                # remove trailing apostrophes
                while current and current[-1] == "'":
                    current.pop()
                if current:
                    token = "".join(current)
                    if len(token) > 1:
                        tokens.append(token)
                current = []
        except Exception:
            #skip unknown characters
            current = []

    while current and current[-1] == "'":
        current.pop()
    if current:
        token = "".join(current)
        if len(token) > 1:
            tokens.append(token)
        
    return tokens

def process_directory(root_path):
    doc_id_counter = 0
    # Iterate through domains in directory/root_path
    for domain in os.listdir(root_path):
        folder_path = os.path.join(root_path, domain)
        
        # Iterate through each page/file in domain
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    content = data.get("content", "")
                    
                    soup = BeautifulSoup(content, "html.parser")
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
    with open(REPORT, "w") as f:
        doc_id_counter = process_directory('test_developer')
        
        size_kb = save_index('inverted_index.json')

        print(f"Documents Indexed: {doc_id_counter}", file=f)
        print(f"Unique Tokens: {len(unique_tokens)}", file=f)
        print(f"Index Size: {size_kb:.2f} KB", file=f)

if __name__ == "__main__":
    generate_report()