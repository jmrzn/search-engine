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
    sorted_index = {k: local_index[k] for k in sorted(local_index)}
    with open(path, 'w') as f:
        json.dump(sorted_index, f)
    print(f"  Flushed partial index #{partial_index_num} ({len(sorted_index)} terms) -> {path}")
    return path

def merge_partial_indexes(partial_files, output_file, offsets_file):
    iterators = []
    for path in partial_files:
        with open(path, 'r') as f:
            data = json.load(f)
        iterators.append(iter(sorted(data.items())))

    firsts = []
    for i, it in enumerate(iterators):
        try:
            firsts.append([next(it), i, it])
        except StopIteration:
            pass

    offsets = {}

    with open(output_file, 'w') as out:
        while firsts:
            min_term = min(f[0][0] for f in firsts)

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

            offsets[min_term] = out.tell()
            out.write(json.dumps({min_term: merged_postings}) + "\n")

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

def get_main_content(soup):
    # Remove boilerplate tags entirely
    for tag in soup.find_all(['nav', 'header', 'footer', 'script', 'style', 'meta', 'link']):
        tag.decompose()

    # Prefer paragraph/article/section text for hashing — avoids shared nav
    # text that survives tag removal (e.g. site titles, breadcrumbs) polluting
    # the SimHash and causing false-positive duplicate matches.
    content_tags = soup.find_all(['p', 'article', 'section', 'main', 'h1', 'h2', 'h3'])
    if content_tags:
        return ' '.join(t.get_text(separator=' ', strip=True) for t in content_tags)

    # Fall back to full page text if no content tags found
    return soup.get_text(separator=' ', strip=True)

def process_directory(root_path):
    doc_id_counter = 0

    partial_index_num = 0
    local_index = defaultdict(list)
    partial_files = []
    local_size = 0

    # (Simhash, filepath, content_length) tuples for near-dup checking
    seen_hashes = []
    SIMHASH_THRESHOLD = 3
    skipped = 0
    total = 0
    ADDED_PREVIEW_LIMIT = 5
    SIDE_BY_SIDE_START = 600
    SIDE_BY_SIDE_END = 650

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
                    soup = parse_content(content)
                    clean_text = soup.get_text()
                    main_content = get_main_content(soup)
                    total += 1

                    try:
                        sh = Simhash(main_content)
                        matched_original = None
                        matched_distance = None
                        matched_content = None
                        for seen_sh, seen_path, seen_content in seen_hashes:
                            d = sh.distance(seen_sh)
                            if d <= SIMHASH_THRESHOLD:
                                matched_original = seen_path
                                matched_distance = d
                                matched_content = seen_content
                                break

                        if matched_original is not None:
                            skipped += 1
                            print(f"Skipped duplicate ({skipped} skipped / {total} total) [distance={matched_distance}]")
                            print(f"  SKIP    ({len(main_content):>6} chars): {file_path}")
                            print(f"  MATCHED ({len(matched_content):>6} chars): {matched_original}")
                            if SIDE_BY_SIDE_START <= skipped <= SIDE_BY_SIDE_END:
                                skip_preview    = main_content[:300].replace('\n', ' ')
                                matched_preview = matched_content[:300].replace('\n', ' ')
                                print(f"  --- skipped content ---")
                                print(f"  {skip_preview}")
                                print(f"  --- matched original content ---")
                                print(f"  {matched_preview}")
                            print()
                            continue
                            print()
                            continue

                        seen_hashes.append((sh, file_path, main_content))
                    except (ValueError, OverflowError):
                        pass  # can't simhash this page, index it anyway

                    tokens = tokenize_text(clean_text)
                    add_to_index(doc_id_counter, tokens, local_index)
                    if doc_id_counter < ADDED_PREVIEW_LIMIT:
                        print(f"Adding doc {doc_id_counter}: {file_path}")
                        print(f"  Content preview ({len(main_content)} chars): {main_content[:300]}\n")
                    else:
                        print(f"Adding doc {doc_id_counter} to index")
                    local_size += len(tokens)
                    doc_id_counter += 1

                    if local_size >= THRESHOLD:
                        path = flush_partial_index(local_index, partial_index_num)
                        partial_files.append(path)
                        partial_index_num += 1
                        local_index = defaultdict(list)
                        local_size = 0

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    if local_index:
        path = flush_partial_index(local_index, partial_index_num)
        partial_files.append(path)
        partial_index_num += 1

    print(f"\nSimHash deduplication: {skipped} pages skipped out of {total} total ({total - skipped} indexed)")
    return doc_id_counter, partial_files

def add_to_index(doc_id, tokens, local_index):
    term_freqs = defaultdict(int)
    for token in tokens:
        term_freqs[token] += 1

    for token, count in term_freqs.items():
        local_index[token].append({'docID': doc_id, 'term_freqs': count})

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

    if partial_files:
        size_kb = merge_partial_indexes(partial_files, FINAL_INDEX_FILE, OFFSETS_FILE)
    else:
        return

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
