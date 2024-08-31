from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from interface.cli_interface import CLIInterface

app = Flask(__name__)


@app.route('/query', methods=['POST'])
def query_documents():
    query = request.json['query']
    interface = CLIInterface()
    ranked_docs = interface.run_final(query_text=query)

    return jsonify({"ranked_documents": ranked_docs.tolist()})


if __name__ == '__main__':
    app.run(port=5002)
