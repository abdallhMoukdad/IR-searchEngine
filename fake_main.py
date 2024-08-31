import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import precision_score, recall_score, average_precision_score

def evaluate_ir_system(qrels, retrieved_docs):
    precision_scores = []
    recall_scores = []
    average_precision_scores = []

    for query_id in qrels:
        if query_id in retrieved_docs:
            true_labels = []
            retrieved_labels = []

            # Create the true labels and retrieved labels lists
            for doc_id in qrels[query_id]:
                true_labels.append(qrels[query_id][doc_id])
                retrieved_labels.append(1 if doc_id in retrieved_docs[query_id] else 0)

            # Calculate precision, recall, and average precision
            precision = precision_score(true_labels, retrieved_labels)
            recall = recall_score(true_labels, retrieved_labels)
            avg_precision = average_precision_score(true_labels, retrieved_labels)

            precision_scores.append(precision)
            recall_scores.append(recall)
            average_precision_scores.append(avg_precision)

    # Calculate mean precision, recall, and MAP
    mean_precision = sum(precision_scores) / len(precision_scores)
    mean_recall = sum(recall_scores) / len(recall_scores)
    mean_avg_precision = sum(average_precision_scores) / len(average_precision_scores)

    return mean_precision, mean_recall, mean_avg_precision

    # Example usage
    # Assume retrieved_docs is a dictionary with query IDs as keys and lists of retrieved document IDs as values
    retrieved_docs = {
        "1": ["197beaca-2019-04-18T11:28:59Z-00001-000", "39a42c2d-2019-04-18T18:27:44Z-00000-000"]
        # Add more queries and retrieved documents
    }

    mean_precision, mean_recall, mean_avg_precision = evaluate_ir_system(qrels, retrieved_docs)
    print(f"Mean Precision: {mean_precision}")
    print(f"Mean Recall: {mean_recall}")
    print(f"Mean Average Precision: {mean_avg_precision}")

def load_qrels(file_path):
    qrels = {}

    # Open the qrels file
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)

        for line in file:
            # Split each line into fields
            query_id, corpus_id, score = line.strip().split('\t')
            score = int(score)

            # Initialize a dictionary for the query ID if it doesn't exist
            if query_id not in qrels:
                qrels[query_id] = {}

            # Add the document ID and relevance score to the dictionary
            qrels[query_id][corpus_id] = score

    return qrels


# Example usage
qrels_file_path = 'path_to_qrels_file.txt'
qrels = load_qrels(qrels_file_path)


class InformationRetrievalSystem:
    def __init__(self, qrels_file_path):
        self.vectorizer = TfidfVectorizer()
        self.qrels = load_qrels(qrels_file_path)
        self.documents = []
        self.document_vectors = None

    def load_documents(self, documents):
        self.documents = documents
        self.document_vectors = self.vectorizer.fit_transform([doc['text'] for doc in documents])

    def search(self, query):
        query_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, self.document_vectors).flatten()

        ranked_doc_indices = similarity_scores.argsort()[::-1]
        return [(self.documents[i]['_id'], similarity_scores[i]) for i in ranked_doc_indices]

    def evaluate(self, retrieved_docs):
        mean_precision, mean_recall, mean_avg_precision = evaluate_ir_system(self.qrels, retrieved_docs)
        return mean_precision, mean_recall, mean_avg_precision


# Load and parse the qrels file
qrels_file_path = 'path_to_qrels_file.txt'
qrels = load_qrels(qrels_file_path)

# Initialize IR system
ir_system = InformationRetrievalSystem(qrels_file_path)

# Load sample documents
sample_documents = [
    {"_id": "doc1", "text": "This is a sample document about Apple."},
    {"_id": "doc2", "text": "This is another sample document about Banana."}
]

ir_system.load_documents(sample_documents)

# Search for a query
query = "Apple"
retrieved_docs = {"1": [doc_id for doc_id, score in ir_system.search(query)]}

# Evaluate the IR system
mean_precision, mean_recall, mean_avg_precision = ir_system.evaluate(retrieved_docs)
print(f"Mean Precision: {mean_precision}")
print(f"Mean Recall: {mean_recall}")
print(f"Mean Average Precision: {mean_avg_precision}")






#---------------------------------------------------------------------------
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class InformationRetrievalSystem:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None
        self.documents = []
        self.queries = {}

    def load_documents(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                self.documents.append(json.loads(line))

    def index_documents(self):
        texts = [doc['text'] for doc in self.documents]
        self.document_vectors = self.vectorizer.fit_transform(texts)

    def load_queries(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                query = json.loads(line)
                self.queries[query['_id']] = query['text']

    def load_qrels(self, file_path):
        qrels = pd.read_csv(file_path, sep='\t', names=["query_id", "corpus_id", "score"])
        return qrels

    def search(self, query_text):
        query_vector = self.vectorizer.transform([query_text])
        similarity_scores = cosine_similarity(query_vector, self.document_vectors).flatten()
        ranked_doc_indices = similarity_scores.argsort()[::-1]
        return [(self.documents[i]['_id'], similarity_scores[i]) for i in ranked_doc_indices]

    def evaluate(self, query_id, retrieved_docs, qrels):
        relevant_docs = qrels[qrels['query_id'] == query_id]
        relevant_doc_ids = relevant_docs[relevant_docs['score'] > 0]['corpus_id'].tolist()
        retrieved_doc_ids = [doc_id for doc_id, _ in retrieved_docs]

        tp = len(set(relevant_doc_ids).intersection(retrieved_doc_ids))
        precision = tp / len(retrieved_doc_ids) if retrieved_doc_ids else 0
        recall = tp / len(relevant_doc_ids) if relevant_doc_ids else 0

        return precision, recall

# Load Data
ir_system = InformationRetrievalSystem()
ir_system.load_documents('path_to_corpus.jsonl')
ir_system.index_documents()
ir_system.load_queries('path_to_queries.jsonl')
qrels = ir_system.load_qrels('path_to_qrels.txt')

# Example of Searching and Evaluating
query_id = '1'
query_text = ir_system.queries[query_id]
retrieved_docs = ir_system.search(query_text)
precision, recall = ir_system.evaluate(query_id, retrieved_docs, qrels)

print(f"Precision: {precision}, Recall: {recall}")
