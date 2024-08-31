import os
import signal
import sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from query.query_processor import QueryProcessor
from storage.storage_manager import StorageManager
import json


class Indexer:
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.word_list = []
        self.word_set = set('test')
        self.processed_docs = []
        self.vectorizer = None  # Hold a reference to the TF-IDF vectorizer
        self.document_vectors = None  # Hold document vectors after transformation
        self.storage_manager = StorageManager()
        self.query_processor = QueryProcessor()
        self.stop_requested = False
        self.current_position = 0  # Initialize current_position
        # Set up signal handling
        # signal.signal(signal.SIGINT, self.handle_stop_signal)
        # signal.signal(signal.SIGTERM, self.handle_stop_signal)

    def handle_stop_signal(self, signum, frame):
        print(f"Signal {signum} received, saving checkpoint and exiting...")
        self.save_intermediate_results()
        self.save_checkpoint(self.current_position)
        self.stop_requested = True

    def save_checkpoint(self, position):
        checkpoint = {
            'position': position,
            'vectorizer': self.vectorizer,
            'inverted_index': self.inverted_index,
            'word_list': list(self.word_set)
        }
        print("Size of your_variable:", sys.getsizeof(self.vectorizer), "bytes")

        self.storage_manager.save_checkpoint(checkpoint)

    def index_documents_from_file_with_stop_signal(self, file_path, batch_size=1000, save_interval=5):
        checkpoint = self.storage_manager.load_checkpoint()
        start_position = 0
        if checkpoint:
            start_position = checkpoint['position']
            self.vectorizer = checkpoint['vectorizer']
            self.inverted_index = checkpoint['inverted_index']
            self.word_set = set(checkpoint['word_list'])
            print(checkpoint['vectorizer'])
            print("Size of your_variable:", sys.getsizeof(start_position), "bytes")
            print("Size of your_variable:", sys.getsizeof(self.vectorizer), "bytes")
            print("Size of your_variable:", sys.getsizeof(self.inverted_index), "bytes")
            print("Size of your_variable:", sys.getsizeof(self.word_set), "bytes")
        self.current_position = start_position  # Set current_position to start_position
        total_size = os.path.getsize(file_path)
        with open(file_path, 'r') as f:
            if start_position > 0:
                f.seek(start_position)
            documents = []
            batch_id = start_position // batch_size
            char_count = start_position
            for line in f:
                if self.stop_requested:
                    break
                char_count += len(line)
                documents.append(json.loads(line))
                if len(documents) >= batch_size:
                    self.process_batch(documents, batch_id)
                    documents = []
                    batch_id += 1

                    # Update checkpoint
                    self.current_position = char_count
                    self.save_checkpoint(self.current_position)

                    if batch_id % save_interval == 0:
                        self.save_intermediate_results()

                    progress = (char_count / total_size) * 100
                    print(f"Progress: {progress:.2f}%")

            if documents and not self.stop_requested:
                self.process_batch(documents, batch_id)
                self.save_intermediate_results()
                print('done')

            if not self.stop_requested:
                self.save_final_results()

    def index_documents(self, documents):
        # Create a TF-IDF vectorizer and fit it to the documents
        self.vectorizer = TfidfVectorizer()
        # Tokenize and preprocess documents
        preprocessed_documents = [
            self.query_processor.complete_process_query(doc.get('title').lower() + doc.get('text').lower()) for doc in
            documents]
        processed_documents = [' '.join(doc) for doc in preprocessed_documents]
        print(preprocessed_documents)
        print(processed_documents)
        self.document_vectors = self.vectorizer.fit_transform(processed_documents)
        # # Save the vectorizer and document vectors
        self.storage_manager.save_vectorizer(self.vectorizer)
        self.storage_manager.save_document_vectors(self.document_vectors)
        for doc_id, document in enumerate(processed_documents):
            # document_words_list = self.query_processor.complete_process_query(document)
            # self.document_vectors = self.vectorizer.fit_transform(document_words_list)
            # self.storage_manager.save_vectorizer(self.vectorizer)
            # self.storage_manager.save_document_vectors(self.document_vectors)

            words = document.lower().split()  # Basic tokenization and lowercasing

            processed_documents = document.lower().split()  # Basic tokenization and lowercasing
            # word_list.append(processed_documents)
            # self.storage_manager.save_vocabulary(word_list)

            for word in processed_documents:
                self.word_list.append(word)

                print('\n------------------------' + word + '\n----------------------------')
                if doc_id not in self.inverted_index[word]:
                    self.inverted_index[word].append(doc_id)
        # Save the inverted index
        self.storage_manager.save_inverted_index(self.inverted_index)
        self.storage_manager.save_vocabulary(self.word_list)

    def load_data(self):
        # Load the inverted index, document vectors, and vectorizer from files
        self.inverted_index = self.storage_manager.load_inverted_index()
        self.vectorizer = self.storage_manager.load_vectorizer()
        self.document_vectors = self.storage_manager.load_document_vectors()

    def index_documents_from_file(self, file_path, batch_size=1000, save_interval=5):
        checkpoint = self.storage_manager.load_checkpoint()
        # checkpoint = self.load_checkpoint()
        start_position = 0
        if checkpoint:
            start_position = checkpoint['position']
            self.vectorizer = checkpoint['vectorizer']
            self.inverted_index = checkpoint['inverted_index']
            self.word_set = set(checkpoint['word_list'])

        # Calculate the total file size in bytes
        total_size = os.path.getsize(file_path)
        with open(file_path, 'r') as f:
            if start_position > 0:
                f.seek(start_position)
            # with open(file_path, 'r') as f:
            #     f.seek(start_position)
            documents = []
            batch_id = start_position // batch_size
            char_count = start_position  # Start tracking char count from the last position
            for line in f:
                char_count += len(line)
                documents.append(json.loads(line))
                if len(documents) >= batch_size:

                    self.process_batch(documents, batch_id)
                    documents = []
                    batch_id += 1

                    # Update checkpoint
                    position = char_count
                    checkpoint = {
                        'position': position,
                        'vectorizer': self.vectorizer,
                        'inverted_index': self.inverted_index,
                        'word_list': self.word_set
                    }

                    self.storage_manager.save_checkpoint(checkpoint)

                    if batch_id % save_interval == 0:
                        self.save_intermediate_results()

                    # Calculate and display progress
                    progress = (char_count / total_size) * 100
                    print(f"Progress: {progress:.2f}%")

            # Process any remaining documents
            if documents:
                self.process_batch(documents, batch_id)
                self.save_intermediate_results()

            # Final save
            self.save_final_results()

    def process_batch(self, documents, batch_id):
        document_ids = [doc['_id'] for doc in documents]

        # Weight factor for the titles
        title_weight = 4
        # self.processed_docs
        for doc in documents:
            doc_id = doc.get('_id')  # Assuming '_id' is the key for document ID
            processed_text = self.query_processor.complete_process_query(
                title_weight * (doc.get('title', ' ').lower() + " ") + doc.get('text', '').lower()
            )

            processed_text = ' '.join(processed_text)
            self.processed_docs.append((doc_id, processed_text))

        # preprocessed_documents = [
        #     zip(doc.get('_id'), self.query_processor.complete_process_query(
        #         title_weight * (doc.get('title', ' ').lower() + " ") + doc.get('text', '').lower()))
        #     for doc in documents
        #
        #     # self.query_processor.complete_process_query(
        #     #     title_weight * (doc.get('title', ' ').lower() + " ") + doc.get('text', '').lower())
        #     # for doc in documents
        # ]
        # print(preprocessed_documents)
        # for z in preprocessed_documents[:1]:
        #     print("this is the z", z)
        #     for idd, text in z:
        #         ' '.join(text)
        #         print("this the id:", idd)
        #         print("this the text:", text)
        # retrieved_ids = [print(id) for id,text in doc_id  doc_id for doc_id in preprocessed_documents[:50]]
        # print(retrieved_ids)
        processed_documents = []
        for doc_id, processed_text in self.processed_docs:
            processed_documents.append(processed_text)
            # Access each value
            # print(f'Document ID: {doc_id}')
            # print(f'Processed Text: {processed_text}')
            # processed_documents = [' '.join(doc) for doc_id, doc in preprocessed_documents]

        if not self.vectorizer:
            self.vectorizer = TfidfVectorizer()
            print('created again')
            # self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=10000)
            batch_vectors = self.vectorizer.fit_transform(processed_documents)
        else:
            batch_vectors = self.vectorizer.transform(processed_documents)
            # new_vectors = self.vectorizer.transform(processed_documents)
            # self.document_vectors = np.vstack([self.document_vectors, new_vectors])
        # self.processed_docs = processed_documents
        # print(self.processed_docs.count())
        # self.processed_docs.
        # self.storage_manager.save_vectorizer(self.vectorizer)
        self.storage_manager.save_processed_docs(self.processed_docs)
        self.processed_docs.clear()
        # self.processed_docs.clear()
        self.storage_manager.save_batch_document_vectors_with_id(batch_vectors, batch_id, document_ids)
        # self.storage_manager.save_batch_document_vectors(batch_vectors, batch_id)
        for doc_id, document in zip(document_ids, processed_documents):  # Use unique document IDs
            words = document.split()
            self.word_set.update(words)

            for word in words:
                if doc_id not in self.inverted_index[word]:
                    self.inverted_index[word].append(doc_id)

        processed_documents.clear()
        document_ids.clear()

        # for doc_id, document in enumerate(processed_documents):
        #     words = document.split()
        #     # self.word_list.extend(words)
        #     self.word_set.update(words)  # Use set to ensure unique words
        #     # print(self.word_set)
        #     # print(type(self.word_set))
        #
        #     for word in words:
        #         if doc_id not in self.inverted_index[word]:
        #             # self.inverted_index[word].append(batch_id * self.storage_manager.batch_size + doc_id)

    def load_all_components(self):
        self.vectorizer = self.storage_manager.load_vectorizer()
        # self.document_vectors = self.storage_manager.load_all_document_vectors()
        self.document_vectors = self.storage_manager.load_all_document_vectors_with_id()
        self.inverted_index = self.storage_manager.load_inverted_index()
        self.word_set = self.storage_manager.load_vocabulary()

    def save_intermediate_results(self):
        self.storage_manager.save_vectorizer(self.vectorizer)
        self.storage_manager.save_processed_docs(self.processed_docs)
        self.storage_manager.save_inverted_index(self.inverted_index)
        self.storage_manager.save_vocabulary(list(self.word_set))
        self.processed_docs.clear()
        print('ineter')

    def save_final_results(self):
        self.storage_manager.save_vectorizer(self.vectorizer)
        self.storage_manager.save_processed_docs(self.processed_docs)

        self.storage_manager.save_inverted_index(self.inverted_index)
        self.storage_manager.save_vocabulary(list(self.word_set))
        self.processed_docs.clear()

        # Optionally remove checkpoint after final save
        if os.path.exists(self.storage_manager.checkpoint_file):
            os.remove(self.storage_manager.checkpoint_file)

    def search(self, query):
        query_words = query.split()
        if not query_words:
            return []

        # Initialize result set with documents containing the first query word
        result_set = set(self.inverted_index.get(query_words[0], []))

        # Intersect results with other query words
        for word in query_words[1:]:
            if word in self.inverted_index:
                result_set.intersection_update(self.inverted_index[word])
            else:
                return []

        return list(result_set)

    def search_vectors(self, query_vector):
        # Calculate cosine similarity between query vector and document vectors
        similarity_scores = cosine_similarity(query_vector, self.document_vectors)

        # Convert similarity scores to a flat list
        similarity_scores = similarity_scores.flatten()

        # Return similarity scores
        return similarity_scores

    def search_vectors_ev(self, query_vector, document_vectors):
        # Calculate cosine similarity between query vector and document vectors
        similarity_scores = cosine_similarity(query_vector, document_vectors)

        # Convert similarity scores to a flat list
        similarity_scores = similarity_scores.flatten()

        # Return similarity scores
        return similarity_scores
