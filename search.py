import os
import json
import math
from datetime import datetime
from nltk import download

download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

INDEX_FILE = "inverted_index.json"
OFFSETS_FILE = "index_offsets.json"
DOC_LENGTHS_FILE = "doc_lengths.json"
DEV_FOLDER = "DEV"
REPORT = "m3_report.txt"

DOC_URL_MAP_FILE = "doc_url_map.json"

def build_url_map():
    with open(DOC_URL_MAP_FILE, 'r') as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

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

def rank_by_tfidf(search_result, query_terms, index, idf, doc_lengths):
    scores = {}
    # search_result_set = set(search_result)

    # print("len(search_result):", len(search_result))
    for term in query_terms:
        if term not in index:
            continue
        # print(term, len(index[term]))

        for posting in index[term]:
            docID = posting['docID']
            # if docID not in search_result_set:
            #     continue

            # Calculate TF-IDF score
            tf = 1 + math.log(posting["term_freqs"])
            tfidf = tf * idf.get(term, 0)

            # Calculate boost for term being in the title/headers/bold
            boost = (
                5 * math.log(1 + posting.get("title_count", 0)) +
                4 * math.log(1 + posting.get("h1_count", 0)) +
                3 * math.log(1 + posting.get("h2_count", 0)) +
                2 * math.log(1 + posting.get("h3_count", 0)) +
                1 * math.log(1 + posting.get("bold_count", 0))
            )

            # Final score
            score = tfidf * (1 + boost)
            scores[docID] = scores.get(docID, 0) + score

    # Length normalization
    for docID in scores:
        scores[docID] /= math.sqrt(doc_lengths.get(docID, 1))

    ranked_result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_result

def get_postings(term, index_file, offsets):
    if term not in offsets:
        return []

    with open(index_file, 'r') as f:
        f.seek(offsets[term])
        line = f.readline()
        entry = json.loads(line)
        return entry[term]

def create_query_index(query_terms, offsets):
    query_index = {}
    for term in query_terms:
        postings = get_postings(term, INDEX_FILE, offsets)
        if postings:
            query_index[term] = postings
    return query_index

def generate_report():
    with open(OFFSETS_FILE, 'r') as f:
        offsets = json.load(f)

    with open(DOC_LENGTHS_FILE, 'r') as d:
        doc_lengths = json.load(d)
    doc_lengths = {int(docID): length for docID, length in doc_lengths.items()}

    url_map = build_url_map()
    queries = ["cristina lopes", "machine learning", "ACM", "master of software engineering"]

    with open(REPORT, "w") as r:
        for query in queries:
            start_time = datetime.now()
            query_terms = process_query(query)
            query_index = create_query_index(query_terms, offsets)
            idf = compute_idf(query_index, len(url_map))

            result = boolean_and_search(query_terms, query_index)
            ranked_result = rank_by_tfidf(result, query_terms, query_index, idf, doc_lengths)
            end_time = datetime.now()

            time_diff = (end_time - start_time).total_seconds() * 1000
            print("\nSearch engine took", time_diff, "ms", file=r)

            print(f"Query: {query}", file=r)
            print(f"Top 5 URLs:", file=r)
            for doc_id in result[:5]:
                print(url_map[doc_id], file=r)

def search():
    with open(OFFSETS_FILE, 'r') as f:
        offsets = json.load(f)

    with open(DOC_LENGTHS_FILE, 'r') as d:
        doc_lengths = json.load(d)
    doc_lengths = {int(docID): length for docID, length in doc_lengths.items()}

    url_map = build_url_map()
    # print(url_map[54462])

    # version that uses console input for queries rather than a set list
    while True:
        query = input("\nEnter a search query or type 'q' to quit: ").strip().lower()
        if (query == "q" or query == "quit"):
            break

        start_time = datetime.now()
        query_terms = process_query(query)
        t1 = datetime.now()

        query_index = create_query_index(query_terms, offsets)
        t2 = datetime.now()

        idf = compute_idf(query_index, len(url_map))
        t3 = datetime.now()

        result = boolean_and_search(query_terms, query_index)
        t4 = datetime.now()

        ranked_result = rank_by_tfidf(result, query_terms, query_index, idf, doc_lengths)
        t5 = datetime.now()
        end_time = datetime.now()

        time_diff = (end_time - start_time).total_seconds() * 1000
        print("Search engine took", time_diff, "ms")
        print(f"process_query: {(t1 - start_time).total_seconds() * 1000:.2f} ms")
        print(f"create_query_index: {(t2 - t1).total_seconds() * 1000:.2f} ms")
        print(f"compute_idf: {(t3 - t2).total_seconds() * 1000:.2f} ms")
        print(f"boolean_and_search: {(t4 - t3).total_seconds() * 1000:.2f} ms")
        print(f"rank_by_tfidf: {(t5 - t4).total_seconds() * 1000:.2f} ms")
        print(f"TOTAL: {(t5 - start_time).total_seconds() * 1000:.2f} ms")

        print(f"\nQuery: {query}")
        print(f"Top 5 URLs:")
        for doc_id, score in ranked_result[:5]:
            print(url_map[doc_id])

if __name__ == "__main__":
    search()
