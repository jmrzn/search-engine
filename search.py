import os
import json
from nltk import download

download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

INDEX_FILE = "inverted_index.json"
DEV_FOLDER = "DEV"
REPORT = "m2_report.txt"

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

def generate_report():
    with open(INDEX_FILE, 'r') as f:
        inverted_index = json.load(f)

    url_map = build_url_map()
    queries = ["cristina lopes", "machine learning", "ACM", "master of software engineering"]

    with open(REPORT, "w") as r:
        for query in queries:
            result = boolean_and_search(process_query(query), inverted_index)
            print(f"\nQuery: {query}", file=r)
            print(f"Top 5 URLs:", file=r)
            for doc_id in result[:5]:
                print(url_map[doc_id], file=r)
                
    # version that uses console input for queries rather than a set list
    # with open(REPORT, "w") as r:
    #     while True:
    #         query = input("Enter a search query or type 'q' to quit: ").strip().lower()
    #         if (query == "q" or query == "quit"):
    #             break
    #         result = boolean_and_search(process_query(query), inverted_index)
    #         print(f"\nQuery: {query}", file=r)
    #         print(f"Top 5 URLs:", file=r)
    #         for doc_id in result[:5]:
    #             print(url_map[doc_id], file=r)
    #         r.flush() # write buffered content to report immediately

if __name__ == "__main__":
    generate_report()
