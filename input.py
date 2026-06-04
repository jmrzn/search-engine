from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json, math
from datetime import datetime
from search import process_query, create_query_index, boolean_and_search, rank_by_tfidf, compute_idf, build_url_map, OFFSETS_FILE, DOC_LENGTHS_FILE

# happens once before flask runs the app to search
with open(OFFSETS_FILE, 'r') as f:
    offsets = json.load(f)
with open(DOC_LENGTHS_FILE, 'r') as d:
    doc_lengths = json.load(d)
    doc_lengths = {int(docID): length for docID, length in doc_lengths.items()}
url_map = build_url_map()

app = Flask(__name__)
CORS(app)

@app.route("/search")
def search_route():
    # gets the query (stuff following ?q=) from the html
    query = request.args.get("q", "").strip().lower()
    print(query)
    if not query:
        return jsonify({"urls": []})

    start_time = datetime.now()
    
    query_terms = process_query(query)
    query_index = create_query_index(query_terms, offsets)
    
    idf = compute_idf(query_index, len(url_map))
    result = boolean_and_search(query_terms, query_index)
    ranked_result = rank_by_tfidf(result, query_terms, query_index, idf, doc_lengths)
    
    end_time = datetime.now()
    time_diff = (end_time - start_time).total_seconds() * 1000
    print("Search engine took", time_diff, "ms")
    print(f"\nQuery: {query}")
    
    urls = [url_map[doc_id] for doc_id, score in ranked_result[:5]]
    return jsonify({"urls": urls})

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/style.css")
def styles():
    return send_from_directory(".", "style.css")

if __name__ == "__main__":
    app.run(port=5000)