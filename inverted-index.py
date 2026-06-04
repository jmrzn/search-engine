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
doc_lengths = {}
REPORT = "report.txt"
DEV_FOLDER = "DEV"
stemmer = PorterStemmer()

THRESHOLD = 500000
PARTIAL_INDEX_DIR = "partial_indexes"
FINAL_INDEX_FILE = "inverted_index.json"
OFFSETS_FILE = "index_offsets.json"
DOC_LENGTHS_FILE = "doc_lengths.json"

def flush_partial_index(local_index, partial_index_num):
    os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
    path = os.path.join(PARTIAL_INDEX_DIR, f"partial_{partial_index_num}.json")
    # sort terms before writing so merging is easier later
    sorted_index = {k: local_index[k] for k in sorted(local_index)}
    with open(path, 'w') as f:
        json.dump(sorted_index, f)
    print(f"  Flushed partial index #{partial_index_num} ({len(sorted_index)} terms) -> {path}")
    return path

def merge_partial_indexes(partial_files, output_file, offsets_file):
    # load each partial file into memory to get iterators in sorted order
    iterators = []
    for path in partial_files:
        with open(path, 'r') as f:
            data = json.load(f)
        iterators.append(iter(sorted(data.items())))
 
    # get the first entry from each partial index for merging
    firsts = [] 
    for i, it in enumerate(iterators):
        try:
            firsts.append([next(it), i, it])
        except StopIteration:
            pass
 
    offsets = {}
 
    with open(output_file, 'w') as out:
        while firsts:
            # find the smallest term (lexicographically)
            min_term = min(f[0][0] for f in firsts)
 
            # get postings from all iterators that have the same min_term
            merged_postings = []
            new_firsts = []
            for entry in firsts:
                (term, postings), idx, it = entry
                if term == min_term:
                    merged_postings.extend(postings)
                    try:
                        new_firsts.append([next(it), idx, it])
                    except StopIteration:
                        pass
                else:
                    new_firsts.append(entry)
            firsts = new_firsts
 
            # record byte offset + write this term's line to the index json file
            offsets[min_term] = out.tell()
            out.write(json.dumps({min_term: merged_postings}) + "\n")
    
    # save the offsets for later when searching
    with open(offsets_file, 'w') as f:
        json.dump(offsets, f)
 
    size_kb = os.path.getsize(output_file) / 1024
    print(f"Final index written to {output_file} ({size_kb:.2f} KB)")
    print(f"Offsets written to {offsets_file} ({len(offsets)} terms)")
    return size_kb

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

def extract_tag_counts(soup):
    title_counts = defaultdict(int)
    h1_counts = defaultdict(int)
    h2_counts = defaultdict(int)
    h3_counts = defaultdict(int)
    bold_counts = defaultdict(int)

    # Extract tokens from the title and count their frequency
    if soup.title:
        for token in tokenize_text(soup.title.get_text(" ", strip=True)):
            title_counts[token] += 1

    # Extract tokens from all H1 headers and count their frequency
    for tag in soup.find_all("h1"):
        for token in tokenize_text(tag.get_text(" ", strip=True)):
            h1_counts[token] += 1

    # Extract tokens from all H2 headers and count their frequency
    for tag in soup.find_all("h2"):
        for token in tokenize_text(tag.get_text(" ", strip=True)):
            h2_counts[token] += 1

    # Extract tokens from all H3 headers and count their frequency
    for tag in soup.find_all("h3"):
        for token in tokenize_text(tag.get_text(" ", strip=True)):
            h3_counts[token] += 1

    # Extract tokens from all bold words and count their frequency
    for tag in soup.find_all(["b", "strong"]):
        for token in tokenize_text(tag.get_text(" ", strip=True)):
            bold_counts[token] += 1

    return title_counts, h1_counts, h2_counts, h3_counts, bold_counts

def process_directory(root_path):
    doc_id_counter = 0
    
    # variables for partial indexing
    partial_index_num = 0
    local_index = defaultdict(list)
    partial_files = []
    local_size = 0
 
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

                    body_tokens = tokenize_text(clean_text)
                    title_counts, h1_counts, h2_counts, h3_counts, bold_counts = extract_tag_counts(soup)

                    add_to_index(doc_id_counter, body_tokens, title_counts, h1_counts, h2_counts, h3_counts, bold_counts, local_index)
                    # print(f"Added doc {doc_id_counter} to index")

                    local_size += len(body_tokens)
                    doc_id_counter += 1

                    # flushes partial index when reach threshold
                    if local_size >= THRESHOLD:
                        path = flush_partial_index(local_index, partial_index_num)
                        partial_files.append(path)
                        partial_index_num += 1
                        local_index = defaultdict(list)
                        local_size = 0
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    # flushes anything left in the local_index
    if local_index:
        path = flush_partial_index(local_index, partial_index_num)
        partial_files.append(path)
        partial_index_num += 1
    
    return doc_id_counter, partial_files

def add_to_index(doc_id, body_tokens, title_counts, h1_counts, h2_counts, h3_counts, bold_counts, local_index):
    # Calculate body term frequency
    term_freqs = defaultdict(int)
    for token in body_tokens:
        term_freqs[token] += 1

    # Create a set of all unique terms that appear anywhere in the doc
    all_terms = set(term_freqs).union(title_counts, h1_counts, h2_counts, h3_counts, bold_counts)

    # A posting for docID, term frequency, and title/headers/bold term frequency if > 0
    for term in all_terms:
        posting = {'docID': doc_id, 'term_freqs': term_freqs.get(term, 0)}

        if title_counts.get(term, 0):
            posting["title_count"] = title_counts[term]

        if h1_counts.get(term, 0):
            posting["h1_count"] = h1_counts[term]

        if h2_counts.get(term, 0):
            posting["h2_count"] = h2_counts[term]

        if h3_counts.get(term, 0):
            posting["h3_count"] = h3_counts[term]

        if bold_counts.get(term, 0):
            posting["bold_count"] = bold_counts[term]

        local_index[term].append(posting)

    # Store doc length (sum of term frequencies)
    doc_lengths[doc_id] = sum(term_freqs.values())

def save_index(output_file):
    with open(output_file, 'w') as f:
        json.dump(index, f)
    return os.path.getsize(output_file) / 1024

def save_doc_lengths(output_file):
    with open(output_file, 'w') as f:
        json.dump(doc_lengths, f)

def generate_report():
    doc_id_counter, partial_files = process_directory(DEV_FOLDER)
    save_doc_lengths(DOC_LENGTHS_FILE)

    # merge partial indexes
    if partial_files:
        size_kb = merge_partial_indexes(partial_files, FINAL_INDEX_FILE, OFFSETS_FILE)
    else:
        return

    # count unique tokens
    unique_token_count = 0
    with open(OFFSETS_FILE, 'r') as f:
        offsets = json.load(f)
        unique_token_count = len(offsets)

    with open(REPORT, "w") as f:
        print(f"Documents Indexed: {doc_id_counter}", file=f)
        print(f"Unique Tokens: {unique_token_count}", file=f)
        print(f"Index Size: {size_kb:.2f} KB", file=f)

if __name__ == "__main__":
    generate_report()
