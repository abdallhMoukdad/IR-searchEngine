import requests
import json

def index_documents_api():
    response = requests.post('http://localhost:5001/index')
    return response.json()

def query_documents_api(query):
    response = requests.post('http://localhost:5002/query', json={'query': query})
    return response.json()

def evaluate_system_api(actual, predicted, k=10):
    response = requests.post('http://localhost:5003/evaluate', json={'actual': actual, 'predicted': predicted, 'k': k})
    return response.json()

def main():

    # Index documents
    index_response = index_documents_api()
    print("Index Response:", index_response)

    # Query
    query = "first document"
    query_response = query_documents_api(query)
    print("Query Response:", query_response)

    # Evaluate
    actual = ["1"]
    predicted = [str(doc_id) for doc_id in query_response['ranked_documents'][:10]]
    evaluation_response = evaluate_system_api(actual, predicted, k=10)
    print("Evaluation Response:", evaluation_response)

if __name__ == '__main__':
    main()
