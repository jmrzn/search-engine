import os
import json
import math
from nltk import download
from datetime import datetime

download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

INDEX_FILE = "inverted_index.json"
DOC_LENGTHS_FILE = "doc_lengths.json"
DEV_FOLDER = "DEV"
REPORT = "m3_report.txt"

def build_url_map(root_path=DEV_FOLDER):
    doc_id_to_url = {}
    doc_id_counter = 0
 
    for domain in os.listdir(root_path):
        folder_path = os.path.join(root_path, domain)
        if not os.path.isdir(folder_path):
            continue

        # Iterate through each page/file in domain
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    url = data.get("url", "")   # grabs url instead of content
                    doc_id_to_url[doc_id_counter] = url
                    doc_id_counter += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return doc_id_to_url

def process_query(query):
    tokens = word_tokenize(query.lower())
    stemmed_tokens = []
    for t in tokens:
        if t.isalpha():
            stemmed_tokens.append(stemmer.stem(t))
    return stemmed_tokens

def boolean_and_search(query_terms, index):
    if not query_terms:
        return []

    postings = []
    for term in query_terms:
        if term not in index:
            return []
        postings.append(index[term])
    postings.sort(key=len)

    result_set = {p['docID'] for p in postings[0]}
    for posting in postings[1:]:
        result_set &= {p['docID'] for p in posting}
    return list(result_set)

def compute_idf(index, num_docs):
    idfs = {}
    for term, postings in index.items():
        idf = math.log(num_docs / len(postings))
        idfs[term] = idf
    return idfs

def rank_by_tfidf(search_result, query_terms, index, idf):
    scores = {}
    with open(DOC_LENGTHS_FILE, 'r') as f:
        doc_lengths = json.load(f)
    doc_lengths = {int(docID): length for docID, length in doc_lengths.items()}

    for term in query_terms:
        if term not in index:
            continue

        for posting in index[term]:
            docID = posting['docID']
            if docID not in search_result:
                continue

            tf = 1 + math.log(posting["term_freqs"])
            tfidf = tf * idf.get(term, 0)
            scores[docID] = scores.get(docID, 0) + tfidf

    for docID in scores:
        scores[docID] /= doc_lengths.get(docID, 1)

    ranked_result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_result

def generate_report():
    with open(INDEX_FILE, 'r') as f:
        inverted_index = json.load(f)

    url_map = build_url_map()
    idf = compute_idf(inverted_index, len(url_map))
    queries = ["cristina lopes", "machine learning", "ACM", "master of software engineering"]

    with open(REPORT, "w") as r:
        for query in queries:
            start_time = datetime.now()
            query_terms = process_query(query)
            result = boolean_and_search(query_terms, inverted_index)
            ranked_result = rank_by_tfidf(result, query_terms, inverted_index, idf)
            end_time = datetime.now()

            time_diff = (end_time - start_time).total_seconds() * 1000
            print("\nSearch engine took", time_diff, "ms", file=r)

            print(f"Query: {query}", file=r)
            print(f"Top 5 URLs:", file=r)
            for doc_id in result[:5]:
                print(url_map[doc_id], file=r)

def search():
    with open(INDEX_FILE, 'r') as f:
        inverted_index = json.load(f)

    url_map = build_url_map()
    idf = compute_idf(inverted_index, len(url_map))

    # version that uses console input for queries rather than a set list
    while True:
        query = input("\nEnter a search query or type 'q' to quit: ").strip().lower()
        if (query == "q" or query == "quit"):
            break

        start_time = datetime.now()
        query_terms = process_query(query)
        result = boolean_and_search(query_terms, inverted_index)
        ranked_result = rank_by_tfidf(result, query_terms, inverted_index, idf)
        end_time = datetime.now()

        time_diff = (end_time - start_time).total_seconds() * 1000
        print("Search engine took", time_diff, "ms")


        print(f"\nQuery: {query}")
        print(f"Top 5 URLs:")
        for doc_id, score in ranked_result[:5]:
            print(url_map[doc_id])

if __name__ == "__main__":
    search()
