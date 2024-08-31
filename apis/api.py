import random
from collections import defaultdict

from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

from evaluate.evaluate import Evaluate
from indexing.indexer import Indexer
from query.query_processor import QueryProcessor
from ranking.ranker import Ranker
from short_evaluation.short_evaluation import indexer, ranker, query_processor, \
    evaluate_api
from storage.storage_manager import StorageManager

# from short_evaluation.short_evaluation import evaluate

app = Flask(__name__)

# Initialize variables
documents = []
document_vectors = None
vectorizer = TfidfVectorizer()
storage_manager = StorageManager()
query_processor = QueryProcessor()
indexer = Indexer()
ranker = Ranker()

inverted_index = defaultdict(list)
word_list = []
word_set = set()



def index_documents(documents):
    global vectorizer
    global document_vectors
    print(documents)
    vectorizer = TfidfVectorizer()
    preprocessed_documents = [
        query_processor.complete_process_query(3 * doc.get('title').lower() + " " + doc.get('text').lower())
        for doc in documents]
    processed_documents = [' '.join(doc) for doc in preprocessed_documents]
    document_vectors = vectorizer.fit_transform(processed_documents)
    for doc_id, document in enumerate(processed_documents):
        for word in document.lower().split():
            word_list.append(word)
            if doc_id not in inverted_index[word]:
                inverted_index[word].append(doc_id)
    return document_vectors


# Load and preprocess documents
def load_documents(file_path):
    global documents, document_vectors
    documents = read_random_records(file_path=file_path, num_records=100)
    return documents
    # with open(file_path, 'r') as file:
    #     documents = [json.loads(line) for line in file]
    # preprocessed_documents = [doc['text'] for doc in documents]
    # document_vectors = vectorizer.fit_transform(preprocessed_documents)


# Search for a query
def search(query_text):
    query_vector = vectorizer.transform([query_text])
    similarity_scores = cosine_similarity(query_vector, document_vectors).flatten()
    ranked_doc_indices = similarity_scores.argsort()[::-1]
    ranked_docs = [(documents[i]['_id'], similarity_scores[i]) for i in ranked_doc_indices]
    return ranked_docs


# Search for a query
def search(query_text, selected_doc_ids):
    query_vector = vectorizer.transform([query_text])
    if selected_doc_ids:
        selected_indices = [i for i, doc in enumerate(documents) if doc['_id'] in selected_doc_ids]
        document_vectors_subset = document_vectors[selected_indices]
        similarity_scores = cosine_similarity(query_vector, document_vectors_subset).flatten()
        ranked_doc_indices = similarity_scores.argsort()[::-1]
        ranked_docs = [(documents[selected_indices[i]]['_id'], similarity_scores[i]) for i in ranked_doc_indices]
    else:
        similarity_scores = cosine_similarity(query_vector, document_vectors).flatten()
        ranked_doc_indices = similarity_scores.argsort()[::-1]
        ranked_docs = [(documents[i]['_id'], similarity_scores[i]) for i in ranked_doc_indices]
    return ranked_docs


# API endpoint for searching
# POST /search
# {
#     "query": "Should teachers get tenure?"
# }
# [
#     {
#         "_id": "c67482ba-2019-04-18T13:32:05Z-00000-000",
#         "score": 0.5
#     },
#     {
#         "_id": "other_document_id",
#         "score": 0.3
#     },
#     ...
# ]


# API endpoint to get all documents
@app.route('/documents', methods=['GET'])
def get_documents():
    return jsonify(documents)


# API endpoint for searching
# POST /search
# {
#     "query": "Should teachers get tenure?",
#     "selected_doc_ids": ["c67482ba-2019-04-18T13:32:05Z-00000-000", "other_document_id"]
# }

# [
#     {
#         "_id": "c67482ba-2019-04-18T13:32:05Z-00000-000",
#         "score": 0.5
#     },
#     {
#         "_id": "other_document_id",
#         "score": 0.3
#     },
#     ...
# ]
# Sample files (file_id -> file_name)
files = {
    'file1': 'Document Set 1',
    'file2': 'Document Set 2',
    'file3': 'Document Set 3'
}

# Sample queries (query_id -> query_text)
queries = {
    '1': 'Should teachers get tenure?',
    '3': 'Should insider trading be allowed?',
    # Add more queries as needed
}


@app.route('/files', methods=['GET'])
def get_files():
    """Retrieve all available files"""
    filess=load_documents(file_path='/home/abdallh/Documents/webis-touche2020/corpus.jsonl')
    print(filess)
    return jsonify(filess)
    # return jsonify(files)


# @app.route('/evaluate', methods=['POST'])
# def evaluate_query():
#     """Evaluate a query based on selected files"""
#     data = request.json
#     query_text = data.get('query_text')
#     selected_files = data.get('file_ids')
#     k = data.get('k', 10)  # Default k to 10 if not provided
#
#     if not query_text or not selected_files:
#         return jsonify({'error': 'Query text and file IDs are required'}), 400
#
#     # Dummy implementation of evaluation (replace with actual logic)
#     evaluation_metrics = {
#         'precision': 0.75,
#         'recall': 0.65,
#         'MAP': 0.70,
#         'MRR': 0.80
#     }
#     retrieved_docs = [
#         {'id': 'doc1', 'title': 'Document 1', 'text': 'Text of document 1', 'score': 0.9},
#         {'id': 'doc2', 'title': 'Document 2', 'text': 'Text of document 2', 'score': 0.85},
#         # Add more dummy documents as needed
#     ]
#
#     return jsonify({
#         'evaluation_metrics': evaluation_metrics,
#         'retrieved_docs': retrieved_docs
#     })


@app.route('/search/all', methods=['POST'])
def search_endpoint_all():
    from interface.cli_interface import CLIInterface
    interface = CLIInterface()
    # # interface.load_data('/home/abdallh/Documents/webis-touche2020/corpus.jsonl')
    # qrels = interface.load_qrels('/home/abdallh/Documents/webis-touche2020/qrels/test.tsv')
    # interface.load_queries('/home/abdallh/Documents/webis-touche2020/queries.jsonl')
    # # interface.run()
    # # Example of Searching and Evaluating
    # query_id = '35'
    # query_text = interface.queries[query_id]
    # retrieved_docs = interface.search_query_with_id(query_text)
    # precision, recall = interface.evaluate(query_id, retrieved_docs, qrels)

    # print(f"Precision: {precision}, Recall: {recall}")

    query_text = request.json['query']
    retrieved_ids, retrieved_results = interface.run_final(query_text=query_text)
    matching_documents = interface.get_docs_by_ids('/home/abdallh/Documents/webis-touche2020/corpus.jsonl',
                                                   retrieved_ids)
    docs=[]
    # for rank, (doc_id, score) in enumerate(retrieved_results[:20]):
    #     document = matching_documents.get(doc_id)
    #     if document:
    #         docs.append(document)
    #         print(f"Rank {rank + 1}: Document {doc_id}, Similarity Score: {score}")
    #         interface.print_ranked_data(document)
    #     else:
    #         print(f"Rank {rank + 1}: Document {doc_id} not found")

    retrieved_docs = [
        {
            '_id': matching_documents.get(doc_id)['_id'],
            'title': matching_documents.get(doc_id)['title'],
            'text': matching_documents.get(doc_id)['text'],
            'score': score
        }
        for doc_id, score in retrieved_results[:10]
    ]
    print(retrieved_docs)
    # selected_doc_ids = request.json.get('selected_doc_ids', [])
    # ranked_docs = search(query_text, selected_doc_ids)
    return jsonify(retrieved_docs)


def read_records_from_files(file_paths, relevant_document_ids):
    documents = []
    with open(file_paths, 'r') as f:
        for line in f:
            doc = json.loads(line)
            if doc['_id'] in relevant_document_ids:
                documents.append(doc)
    return documents


@app.route('/search/selected', methods=['POST'])
def search_endpoint():
    data = request.json
    query_text = data.get('query_text')
    selected_files = data.get('file_ids')
    k = data.get('k', 10)
    if not query_text or not selected_files:
        return jsonify({'error': 'Query text and file IDs are required'}), 400

    records = read_records_from_files(file_paths='/home/abdallh/Documents/webis-touche2020/corpus.jsonl',
                                      relevant_document_ids=selected_files)
    documents = records
    documents_vectors = index_documents(documents)
    query_text.lower()

    processed_query = query_processor.complete_process_query(query_text)
    joined_string = " ".join(processed_query)

    query_vector = vectorizer.transform([joined_string])
    similarity_scores = indexer.search_vectors_ev(query_vector, documents_vectors)
    retrieved_results = ranker.rank_vectors_results_reutrn_tuples(similarity_scores)
    print(selected_files)
    retrieved_docs = [
        {
            'id': records[doc_id]['_id'],
            'title': records[doc_id]['title'],
            'text': records[doc_id]['text'],
            'score': score
        }
        for doc_id, score in retrieved_results[:k]
    ]
    print(retrieved_docs)

    return jsonify({
        'retrieved_docs': retrieved_docs
    })


@app.route('/evaluate', methods=['POST'])
def evaluate_endpoint():
    data = request.json
    query_id = data.get('query_id')
    query_text = data.get('query_text')
    k = data.get('k', 10)

    if not query_id or not query_text:
        return jsonify({'error': 'Query ID and text are required'}), 400

    evaluation_metrics, retrieved_docs = evaluate_api(query_id, query_text, k)
    response = {
        'evaluation': evaluation_metrics,
        'documents': retrieved_docs
    }
    return jsonify(response)
@app.route('/evaluate/all', methods=['POST'])
def evaluate_all_endpoint():
    data = request.json
    query_id = data.get('query_id')
    query_text = data.get('query_text')
    k = data.get('k', 10)

    if not query_id or not query_text:
        return jsonify({'error': 'Query ID and text are required'}), 400

    evaluation_metrics, retrieved_docs = evaluate_api(query_id, query_text, k)
    response = {
        'evaluation': evaluation_metrics,
        'documents': retrieved_docs
    }
    return jsonify(response)


def read_random_records(file_path, num_records=10):
    # Step 1: Count the total number of lines in the file
    with open(file_path, 'r') as file:
        total_lines = sum(1 for _ in file)
    if num_records > total_lines:
        print(total_lines)
        raise ValueError("num_records is larger than the total number of lines in the file.")
    # Step 2: Randomly select unique line numbers
    random_lines = random.sample(range(total_lines), num_records)

    # Step 3: Read and store the records from these selected lines
    records = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i in random_lines:
                query = json.loads(line)
                records.append(query)
                # records[query['_id']] = query['text']
            if len(records) == num_records:
                break

    return records

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_data = request.json
    doc_id = feedback_data.get('doc_id')
    relevant = feedback_data.get('relevant')
    user_feedback = feedback_data.get('feedback', '')

    if doc_id is None or relevant is None:
        return jsonify({'error': 'doc_id and relevant fields are required'}), 400

    # Store the feedback in a database or file
    with open('feedback.txt', 'a') as f:
        f.write(f"{doc_id}\t{relevant}\t{user_feedback}\n")

    return jsonify({'message': 'Feedback received'}), 200
@app.route('/queries', methods=['GET'])
def get_queries():
    records = read_random_records(file_path='/home/abdallh/Documents/webis-touche2020/queries.jsonl', num_records=14)
    queries = [
        {"id": "1", "text": "Should teachers get tenure?"},
        {"id": "2", "text": "Example query 2"},
        # Add more predefined queries here
    ]
    print(records)
    # return jsonify(queries)
    return jsonify(records)


@app.route('/search/evaluation', methods=['POST'])
def search_endpoint1():
    # evaluate(qrels_file='/home/abdallh/Documents/webis-touche2020/qrels/test.tsv',
    #          queries_file='/home/abdallh/Documents/webis-touche2020/queries.jsonl',
    #          sample_size=2
    #          )

    query_id = request.json['query_id']
    query_text = request.json['query_text']
    # query_eva_result =
    pass
    # documents and evaluation for this query
    selected_doc_ids = request.json.get('selected_doc_ids', [])
    ranked_docs = search(query_text, selected_doc_ids)
    return jsonify(ranked_docs)


# @app.route('/search', methods=['POST'])
# def search_endpoint():
#     query_text = request.json['query']
#     ranked_docs = search(query_text)
#     return jsonify(ranked_docs)


# API endpoint to get document by ID
@app.route('/document/<doc_id>', methods=['GET'])
def get_document(doc_id):
    document = next((doc for doc in documents if doc['_id'] == doc_id), None)
    if document:
        return jsonify(document)
    else:
        return jsonify({'error': 'Document not found'}), 404


# API endpoint to add a new document
@app.route('/document', methods=['POST'])
def add_document():
    new_document = request.json
    documents.append(new_document)
    document_vectors = vectorizer.fit_transform([doc['text'] for doc in documents])
    return jsonify({'message': 'Document added successfully'})


# API endpoint to delete a document by ID
@app.route('/document/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    global documents, document_vectors
    documents = [doc for doc in documents if doc['_id'] != doc_id]
    document_vectors = vectorizer.fit_transform([doc['text'] for doc in documents])
    return jsonify({'message': 'Document deleted successfully'})


# API endpoint to update a document by ID
@app.route('/document/<doc_id>', methods=['PUT'])
def update_document(doc_id):
    document_data = request.json
    for doc in documents:
        if doc['_id'] == doc_id:
            doc.update(document_data)
            break
    document_vectors = vectorizer.fit_transform([doc['text'] for doc in documents])
    return jsonify({'message': 'Document updated successfully'})


@app.route('/')
def hello_world():
    return 'Hello, World!'


# API endpoint to evaluate the IR system
# @app.route('/evaluate', methods=['POST'])
# def evaluate_system():
#     query_id = request.json['query_id']
#     retrieved_docs = request.json['retrieved_docs']
#     qrels = request.json['qrels']
#     # Implement evaluation logic here
#     return jsonify({'message': 'Evaluation completed'})


# API endpoint to reindex documents
@app.route('/reindex', methods=['POST'])
def reindex_documents():
    load_documents('corpus.jsonl')
    return jsonify({'message': 'Documents reindexed successfully'})


# API endpoint to clear index
@app.route('/clear-index', methods=['POST'])
def clear_index():
    global documents, document_vectors
    documents = []
    document_vectors = None
    return jsonify({'message': 'Index cleared successfully'})


# Main function to start the server
if __name__ == '__main__':
    # load_documents('corpus.jsonl')
    app.run(debug=True)
