import json
import random
import sys
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

from evaluate.evaluate import Evaluate
from indexing.indexer import Indexer
from query.query_processor import QueryProcessor
from ranking.ranker import Ranker
from storage.storage_manager import StorageManager

inverted_index = defaultdict(list)
word_list = []
word_set = set('test')

vectorizer = None  # Hold a reference to the TF-IDF vectorizer
document_vectors = None  # Hold document vectors after transformation
storage_manager = StorageManager()
query_processor = QueryProcessor()
indexer = Indexer()
ranker = Ranker()


def index_documents(documents):
    # Create a TF-IDF vectorizer and fit it to the documents
    global vectorizer
    global document_vectors
    global storage_manager
    global query_processor
    global inverted_index
    global word_set
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=10000)
    # vectorizer = TfidfVectorizer()
    # Tokenize and preprocess documents
    # Weight factor for the titles
    title_weight = 3

    preprocessed_documents = [
        query_processor.complete_process_query(title_weight * doc.get('title').lower() + " " + doc.get('text').lower())
        for doc in
        documents]
    processed_documents = [' '.join(doc) for doc in preprocessed_documents]
    # print(preprocessed_documents)
    # print(processed_documents)
    document_vectors = vectorizer.fit_transform(processed_documents)
    # # Save the vectorizer and document vectors
    # self.storage_manager.save_vectorizer(self.vectorizer)
    # self.storage_manager.save_document_vectors(self.document_vectors)
    for doc_id, document in enumerate(processed_documents):
        words = document.lower().split()  # Basic tokenization and lowercasing

        processed_document = document.lower().split()  # Basic tokenization and lowercasing
        # word_list.append(processed_documents)
        # self.storage_manager.save_vocabulary(word_list)

        for word in processed_document:
            word_list.append(word)

            # print('\n------------------------' + word + '\n----------------------------')
            if doc_id not in inverted_index[word]:
                inverted_index[word].append(doc_id)
    # Save the inverted index
    return document_vectors


def evaluate(qrels_file, queries_file, sample_size):
    global vectorizer
    global document_vectors
    global storage_manager
    global query_processor
    global inverted_index
    global word_set

    with open(qrels_file, 'r') as f:
        qrels = f.readlines()

    with open(queries_file, 'r') as f:
        queries = [json.loads(line) for line in f]

    sample_queries = random.sample(queries, sample_size)

    for query in sample_queries:
        query_id = query['_id']
        query_text = query['text']


        relevant_docs = [line.split()[1] for line in qrels if line.split()[0] == query_id]

        text_to_be_processed = read_records_from_corpus(
            corpus_file='/home/abdallh/Documents/webis-touche2020/corpus.jsonl',
            relevant_document_ids=relevant_docs)
        print(len(text_to_be_processed))
        documents_vectors = index_documents(text_to_be_processed)
        processed_query = query_processor.complete_process_query(query_text)
        # Join the list of strings into a single string
        joined_string = " ".join(processed_query)
        print(joined_string)
        # Transform the processed query to VSM using the same vectorizer as the documents
        query_vector = vectorizer.transform([joined_string])
        similarity_scores = indexer.search_vectors_ev(query_vector, documents_vectors)
        retrieved_results = ranker.rank_vectors_results_reutrn_tuples(similarity_scores)
        # print(retrieved_results)

        # retrieved_results = search_engine.search_query(query_text)

        retrieved_docs = [text_to_be_processed[doc_id]['_id'] for doc_id, score in retrieved_results[:10]]
        # print(retrieved_docs)
        # precision = len(set(relevant_docs) & set(retrieved_docs)) / len(retrieved_docs)
        # recall = len(set(relevant_docs) & set(retrieved_docs)) / len(relevant_docs)
        e = Evaluate(actual=relevant_docs, predicted=retrieved_docs, k=10)
        e.print_all()
        # print(f"Query ID: {query_id}")
        # print(f"Precision: {precision}")
        # print(f"Recall: {recall}")
        # print('retrieved', len(retrieved_docs))
        # print('relevant', len(relevant_docs))
        # print_retrieved_vs_right_documents_path(retrieved_docs, relevant_docs,
        # '/home/abdallh/Documents/webis-touche2020/corpus.jsonl')

def evaluate_api(query_id, query_text, k=10):
    with open('/home/abdallh/Documents/webis-touche2020/qrels/test.tsv', 'r') as f:
        qrels = f.readlines()

    relevant_docs = [line.split()[1] for line in qrels if line.split()[0] == query_id]

    text_to_be_processed = read_records_from_corpus(
        corpus_file='/home/abdallh/Documents/webis-touche2020/corpus.jsonl',
        relevant_document_ids=relevant_docs)

    documents_vectors = index_documents(text_to_be_processed)
    processed_query = query_processor.complete_process_query(query_text)
    joined_string = " ".join(processed_query)

    # Transform the processed query to VSM using the same vectorizer as the documents
    query_vector = vectorizer.transform([joined_string])
    similarity_scores = indexer.search_vectors_ev(query_vector, documents_vectors)
    retrieved_results = ranker.rank_vectors_results_reutrn_tuples(similarity_scores)
    print(retrieved_results)
    print("Size of your_variable:", sys.getsizeof(vectorizer), "bytes")

    retrieved_docs = [
        {
            '_id': text_to_be_processed[doc_id]['_id'],
            'title': text_to_be_processed[doc_id]['title'],
            'text': text_to_be_processed[doc_id]['text'],
            'score': score
        }
        for doc_id, score in retrieved_results
    ]
    retrieved_docs_id = [text_to_be_processed[doc_id]['_id'] for doc_id, score in retrieved_results]

    e = Evaluate(actual=relevant_docs, predicted=retrieved_docs_id, k=k)
    e.print_all()
    metrics = e.get_metrics()
    e.print_all()
    evaluation_metrics = {
        'precision_at_k': metrics.get('precision_at_k'),
        'recall': metrics.get('recall'),
        'MAP': metrics.get('MAP'),
        'MRR': metrics.get('MRR')
    }

    return evaluation_metrics, retrieved_docs

def print_retrieved_vs_right_documents(retrieved_docs, right_docs, corpus):
    """
    Print retrieved documents and right documents for comparison.

    Args:
    - retrieved_docs (list): List of retrieved document IDs.
    - right_docs (list): List of document IDs that satisfy the query.
    - corpus (dict): Dictionary containing the corpus data with document IDs as keys.

    Returns:
    None
    """
    print("Retrieved Documents:")
    for doc_id in retrieved_docs:
        if doc_id in corpus:
            document = corpus[doc_id]
            print(f"Document ID: {doc_id}")
            print("Title:", document.get('title'))
            print("Text:", document.get('text'))
            print()

    print("\nRight Documents:")
    for doc_id in right_docs:
        if doc_id in corpus:
            document = corpus[doc_id]
            print(f"Document ID: {doc_id}")
            print("Title:", document.get('title'))
            print("Text:", document.get('text'))
            print()


import json


def print_retrieved_vs_right_documents_path(retrieved_docs, relevant_docs, corpus_file):
    """
    Print the retrieved documents and the relevant documents for comparison.

    Args:
    - retrieved_docs (list): List of document IDs retrieved by the IR system.
    - relevant_docs (list): List of relevant document IDs from the qrels file.
    - corpus_file (str): Path to the corpus file (JSONL format).

    Returns:
    - None
    """
    # Read the corpus from the file
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            corpus[record['_id']] = record

    print("Retrieved Documents:")
    for doc_id in retrieved_docs:
        if doc_id in corpus:
            print(f"Document ID: {doc_id}")
            print(f"Title: {corpus[doc_id]['title']}")
            print(f"Text: {corpus[doc_id]['text']}")
            print()

    print("Relevant Documents:")
    for doc_id in relevant_docs:
        if doc_id in corpus:
            print(f"Document ID: {doc_id}")
            print(f"Title: {corpus[doc_id]['title']}")
            print(f"Text: {corpus[doc_id]['text'][:100]}")
            print()


def read_records_from_corpus(corpus_file, relevant_document_ids):
    """
    Read records from the corpus file, including 100 random records,
    in addition to the records corresponding to the relevant document IDs.

    Args:
    - corpus_file (str): Path to the corpus file (JSONL format).
    - relevant_document_ids (list): List of relevant document IDs to include.

    Returns:
    - list: List of records, including records from relevant document IDs
            and 100 randomly selected records from the corpus file.
    """
    records = []

    # Select 100 random document IDs from the entire corpus
    random_document_ids = set(random.sample(relevant_document_ids, min(len(relevant_document_ids), 100)))

    # Convert relevant_document_ids list to a set
    relevant_document_ids_set = set(relevant_document_ids)

    # Combine relevant_document_ids_set and random_document_ids to create a set of document IDs to read
    document_ids_to_read = relevant_document_ids_set.union(random_document_ids)
    print(len(relevant_document_ids))
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record['_id'] in document_ids_to_read:
                records.append(record)
            if len(records) >= 100 + len(relevant_document_ids):
                break  # Stop reading more records if we have enough
    print(len(records))

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if len(records) <= 500 + len(relevant_document_ids):
                records.append(record)
                # print(len(records))
            else:
                break

    print(len(records))
    return records


# def read_records_from_corpus(corpus_file, relevant_document_ids):
#     """
#     Read records from the corpus file based on the list of relevant document IDs
#     and select 100 random document IDs.
#
#     Args:
#     - corpus_file (str): Path to the corpus file (JSONL format).
#     - relevant_document_ids (list): List of relevant document IDs to read.
#
#     Returns:
#     - list: List of records corresponding to the relevant document IDs and
#             100 randomly selected document IDs.
#     """
#     records = []
#     random_document_ids = random.sample(relevant_document_ids, 100)
#     document_ids_to_read = set(relevant_document_ids + random_document_ids)
#
#     with open(corpus_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             record = json.loads(line)
#             if record['_id'] in document_ids_to_read:
#                 records.append(record)
#     return records


# def read_records_from_corpus(corpus_file, document_ids):
#     """
#     Read records from the corpus file based on the list of document IDs.
#
#     Args:
#     - corpus_file (str): Path to the corpus file (JSONL format).
#     - document_ids (list): List of document IDs to read.
#
#     Returns:
#     - list: List of records corresponding to the document IDs.
#     """
#     records = []
#     with open(corpus_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             record = json.loads(line)
#             if record['_id'] in document_ids:
#                 records.append(record)
#     return records


# # Example usage:
# corpus_file_path = 'path/to/corpus.jsonl'
# document_ids_to_read = ['c67482ba-2019-04-18T13:32:05Z-00000-000', 'c67482ba-2019-04-18T13:32:05Z-00001-000']
# records = read_records_from_corpus(corpus_file_path, document_ids_to_read)
# for record in records:
#     print(record)
def main():
    evaluate(qrels_file='/home/abdallh/Documents/webis-touche2020/qrels/test.tsv',
             queries_file='/home/abdallh/Documents/webis-touche2020/queries.jsonl',
             sample_size=2
             )


if __name__ == '__main__':
    main()
