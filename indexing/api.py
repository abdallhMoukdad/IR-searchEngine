from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

from indexing.indexer import Indexer

app = Flask(__name__)
@app.route('/index', methods=['POST'])
def index_documents():
    # documents = request.json
    indexer = Indexer()
    indexer.index_documents_from_file_with_stop_signal(file_path='/home/abdallh/Documents/webis-touche2020/corpus.jsonl')

    return jsonify({"status": "indexed", "documents_count": len(indexer.processed_docs)})


if __name__ == '__main__':
    app.run(port=5001)