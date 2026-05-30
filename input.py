from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json, math
from search import process_query, boolean_and_search, rank_by_tfidf, compute_idf, build_url_map, INDEX_FILE

app = Flask(__name__)
CORS(app)

@app.route("/search")
def search_route():
    with open(INDEX_FILE, 'r') as f:
        inverted_index = json.load(f)

    url_map = build_url_map()
    idf = compute_idf(inverted_index, len(url_map))

    query = request.args.get("q", "").strip()
    print(query)
    if not query:
        return jsonify({"urls": []})

    query_terms = process_query(query)
    result = boolean_and_search(query_terms, inverted_index)
    ranked_result = rank_by_tfidf(result, query_terms, inverted_index, idf)

    urls = [url_map[doc_id] for doc_id, _ in ranked_result[:5]]
    return jsonify({"urls": urls})

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/style.css")
def styles():
    return send_from_directory(".", "style.css")

if __name__ == "__main__":
    app.run(port=5000)