import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import data.read as r
from query.query_processor import QueryProcessor
from indexing.indexer import Indexer
from ranking.ranker import Ranker
import xml.etree.ElementTree as ET
import json
import pandas as pd

from scipy.sparse import vstack
import re
import ir_datasets
import os
from rank_bm25 import BM25Okapi


class CLIInterface:
    def __init__(self):
        self.queries = {}
        self.query_processor = QueryProcessor()
        self.indexer = Indexer()
        self.ranker = Ranker()
        self.documents = []

    def load_queries(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                query = json.loads(line)
                self.queries[query['_id']] = query['text']

    def load_qrels(self, file_path):
        qrels = pd.read_csv(file_path, sep='\t', names=["query_id", "corpus_id", "score"])
        return qrels

    def load_data(self, file_path):
        # Load the dataset
        # self.documents = self.load_dataset(file_path)
        self.documents = self.read_jsonl_file(file_path)
        # self.process_documents(self.documents)

    def search(self, query_text):
        # self.indexer.index_documents(self.documents)

        # processed_query = self.query_processor.process_query(query)
        processed_query = self.query_processor.complete_process_query(query_text)
        # Transform the processed query to VSM using the same vectorizer as the documents
        query_vector = self.indexer.vectorizer.transform(processed_query)

        # Perform the search to get similarity scores
        similarity_scores = self.indexer.search_vectors(query_vector)
        # Rank the results based on similarity scores
        ranked_results = self.ranker.rank_vectors_results(similarity_scores)
        print(similarity_scores)
        print(ranked_results)
        return [(self.documents[i]['_id'], similarity_scores[i]) for i in ranked_results]
        # return [(self.documents[doc_id]['_id'], similarity_scores[doc_id]) for rank, doc_id in
        #         enumerate(ranked_results)]

    def search2(self, query_text):
        # self.indexer.index_documents(self.documents)
        self.indexer.index_documents_from_file(file_path='/home/abdallh/Documents/webis-touche2020/corpus.jsonl')
        # self.indexer.load_data()
        query_text.lower()
        query_vector = self.indexer.vectorizer.transform([query_text])
        similarity_scores = cosine_similarity(query_vector, self.indexer.document_vectors).flatten()
        ranked_doc_indices = similarity_scores.argsort()[::-1]
        print(similarity_scores)
        print(ranked_doc_indices)
        for rank, doc_id in enumerate(ranked_doc_indices):
            # doc_id = int(doc_id)
            # print(f"Rank {rank + 1}: Document {doc_id + 1}: {documents[doc_id]}")
            # print(self.documents[0])
            self.print_ranked_data(self.documents[doc_id])

        return [(self.documents[i]['_id'], similarity_scores[i]) for i in ranked_doc_indices]

    import numpy as np
    def search_query1(self, query_text):

        processed_query = self.query_processor.complete_process_query(query_text)
        query_text = ' '.join(processed_query)
        self.indexer.load_all_components()
        # self.indexer.index_documents_from_file(file_path='/home/abdallh/Documents/webis-touche2020/corpus.jsonl')
        # Load the query vector
        query_vector = self.indexer.vectorizer.transform([query_text])
        # Initialize lists to store similarity scores and document IDs
        similarity_scores = []
        document_ids = []

        # Iterate over each batch of document vectors
        for batch_id in range(self.indexer.storage_manager.get_num_batches()):
            # Load the document vectors for the current batch
            document_vectors = self.indexer.storage_manager.load_batch_document_vectors(batch_id)

            # Calculate cosine similarity between query vector and document vectors
            similarity_batch = cosine_similarity(query_vector, document_vectors)

            # Flatten the similarity matrix to get similarity scores for each document in the batch
            similarity_scores.extend(similarity_batch.flatten())

            # Populate document IDs corresponding to the batch
            document_ids.extend(
                range(batch_id * self.indexer.storage_manager.batch_size,
                      (batch_id + 1) * self.indexer.storage_manager.batch_size))

        # Combine similarity scores with corresponding document IDs
        results = list(zip(document_ids, similarity_scores))

        # Sort the results based on similarity scores in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def search_query(self, query_text):
        processed_query = self.query_processor.complete_process_query(query_text)
        query_text = ' '.join(processed_query)
        self.indexer.load_all_components()
        # self.indexer.index_documents_from_file(file_path='/home/abdallh/Documents/webis-touche2020/corpus.jsonl')

        # Load the query vector
        query_vector = self.indexer.vectorizer.transform([query_text])

        # Initialize lists to store similarity scores and document IDs
        similarity_scores = []
        document_ids = []
        all_document_vectors = []
        # Iterate over each batch of document vectors
        batch_id = 0
        while True:
            try:
                # Load the document vectors for the current batch
                document_vectors = self.indexer.storage_manager.load_batch_document_vectors(batch_id)
                all_document_vectors.extend(document_vectors)
                # Calculate cosine similarity between query vector and document vectors
                similarity_batch = cosine_similarity(query_vector, document_vectors)

                # Flatten the similarity matrix to get similarity scores for each document in the batch
                similarity_scores.extend(similarity_batch.flatten())

                # Populate document IDs corresponding to the batch
                document_ids.extend(
                    range(batch_id * self.indexer.storage_manager.batch_size,
                          (batch_id + 1) * self.indexer.storage_manager.batch_size))

                batch_id += 1
            except FileNotFoundError:
                # If the batch file does not exist, assume we've reached the end
                break

        # Combine similarity scores with corresponding document IDs
        results = list(zip(document_ids, similarity_scores))
        for doc_id, sim_score in results:
            print(f"Document ID: {doc_id}, Similarity Score: {sim_score}")
        print(len(all_document_vectors))
        print(all_document_vectors[1])
        results = [(all_document_vectors[doc_id]['_id'], sim_score) for doc_id, sim_score in
                   zip(document_ids, similarity_scores)]

        # Sort the results based on similarity scores in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def search_query_with_id(self, query_text):
        processed_query = self.indexer.query_processor.complete_process_query(query_text)
        query_text = ' '.join(processed_query)
        # self.indexer.load_all_components()
        # self.indexer.index_documents_from_file(file_path='/home/abdallh/Documents/webis-touche2020/corpus.jsonl')
        self.indexer.index_documents_from_file_with_stop_signal(
            file_path='/home/abdallh/Documents/webis-touche2020/corpus.jsonl')

        # query_vector = self.indexer.vectorizer.transform([query_text])
        # similarity_scores = []
        # document_ids = []
        #
        # for batch_file in os.listdir(self.indexer.storage_manager.document_vectors_dir):
        #     if re.match(r'^document_vectors_\d+\.pkl$', batch_file):
        #         batch_id = int(batch_file.split('_')[2].split('.')[0])
        #         document_vectors, batch_document_ids = self.indexer.storage_manager.load_batch_document_vectors(
        #             batch_id)
        #         similarity_batch = cosine_similarity(query_vector, document_vectors).flatten()
        #
        #         similarity_scores.extend(similarity_batch)
        #         document_ids.extend(batch_document_ids)
        #
        # results = list(zip(document_ids, similarity_scores))
        # results.sort(key=lambda x: x[1], reverse=True)
        #
        # return results

    def search_components(self, query_text):
        # Ensure all components are loaded

        # if not self.indexer.vectorizer or not self.indexer.document_vectors or not self.indexer.inverted_index or not self.indexer.word_set:
        #     self.indexer.load_all_components()

        self.indexer.index_documents_from_file()

        # Process the query
        processed_query = self.query_processor.complete_process_query(query_text)
        query_text = ' '.join(processed_query)
        query_vector = self.indexer.vectorizer.transform([query_text])
        # Perform the search to get similarity scores
        # similarity_scores = self.indexer.search_vectors(query_vector)
        # # Rank the results based on similarity scores
        # ranked_results = self.ranker.rank_vectors_results(similarity_scores)

        # Print the shape of the query vector
        print("Query Vector Shape:", query_vector.shape)
        print(type(self.indexer.document_vectors))  # Output: <class 'numpy.ndarray'>
        for idx, sparse_matrix in enumerate(self.indexer.document_vectors):
            # Perform operations on the sparse matrix
            # For example, print some information about each matrix
            print(f"Matrix {idx + 1}:")
            print(f"Shape: {sparse_matrix.shape}")
            print(f"Number of non-zero elements: {sparse_matrix.nnz}")
            print()

        # Concatenate sparse matrices vertically if needed
        concatenated_matrix = vstack(self.indexer.document_vectors)
        # Check if document vectors are sparse matrices
        if isinstance(self.indexer.document_vectors, list):
            # Convert sparse matrices to dense matrices for better readability
            # Convert list of sparse matrices to a single dense matrix
            document_vectors_dense = np.vstack([batch.toarray() for batch in self.indexer.document_vectors])
            # document_vectors_dense = [batch.toarray() for batch in self.indexer.document_vectors]
        else:
            # If document vectors are already dense, use them directly
            document_vectors_dense = self.indexer.document_vectors
            # document_vectors_dense = np.vstack([batch.toarray() for batch in self.indexer.document_vectors])
        print(type(document_vectors_dense))  # Output: <class 'scipy.sparse.csr.csr_matrix'>
        # Print the shape of the document vectors
        print("Document Vectors Shape:", len(document_vectors_dense))
        print("Document Vectors Shape:", document_vectors_dense.shape)

        # Print the first few elements of the query vector
        print("First few elements of Query Vector:")
        print(query_vector[:10])

        # Print the first few elements of the document vectors
        print("First few elements of Document Vectors:")
        for batch in document_vectors_dense[:10]:
            print(batch)

        # Ensure that query_vector and document_vectors_dense are numpy arrays
        query_vector = np.asarray(query_vector)
        document_vectors_dense = np.asarray(document_vectors_dense)

        # Compute similarity scores
        similarity_scores = cosine_similarity(query_vector, document_vectors_dense).flatten()

        # Rank documents based on similarity scores
        ranked_doc_indices = similarity_scores.argsort()[::-1]

        # Print similarity scores and ranked document indices
        print("Similarity Scores:")
        print(similarity_scores)
        print("Ranked Document Indices:")
        print(ranked_doc_indices)

        # Print ranked documents
        for rank, doc_id in enumerate(ranked_doc_indices):
            self.print_ranked_data(self.documents[doc_id])

        # Return ranked document IDs and similarity scores
        return [(self.documents[i]['_id'], similarity_scores[i]) for i in ranked_doc_indices]

    def run(self):
        print("Welcome to the Information Retrieval System!")

        # Index the sample documents (optional if documents are already indexed)
        # if not self.indexer.document_vectors:
        #     self.indexer.index_documents(documents)
        # Index the sample documents
        # self.indexer.index_documents(documents)
        self.indexer.index_documents(self.documents)
        # self.indexer.load_data()
        while True:
            query = input("\nEnter your search query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            # Process the query
            # processed_query = self.query_processor.process_query(query)
            processed_query = self.query_processor.complete_process_query(query)
            # Join the list of strings into a single string
            joined_string = " ".join(processed_query)
            print(joined_string)
            # Transform the processed query to VSM using the same vectorizer as the documents
            query_vector = self.indexer.vectorizer.transform([joined_string])
            print(type(self.indexer.document_vectors))  # Output: <class 'numpy.ndarray'>
            print(type(query_vector))

            # Perform the search to get similarity scores
            similarity_scores = self.indexer.search_vectors(query_vector)
            print("Query Vector Shape:", query_vector.shape)
            # Print the shape of the document vectors
            print("Document Vectors Shape:", self.indexer.document_vectors.shape[0])
            print("Document Vectors Shape:", self.indexer.document_vectors.shape)

            # Rank the results based on similarity scores
            ranked_results = self.ranker.rank_vectors_results(similarity_scores)
            # processed_query = self.query_processor.process_query(query)
            # results = self.indexer.search(processed_query)
            # Display the ranked search results
            # Check the top similarity score against the threshold
            # top_similarity_score = similarity_scores[0] if similarity_scores.size>0 else 0
            print("this the similarity score:", similarity_scores)
            print(ranked_results)
            # if top_similarity_score ==0:
            #     # If the top score is below the threshold, return an unsure message
            #     print("The system is unsure about the query. No relevant documents found.")
            #     # return "The system is unsure about the query. No relevant documents found."
            # else:
            print("\nSearch Results (ranked by relevance):")
            for rank, doc_id in enumerate(ranked_results[:10]):
                # doc_id = int(doc_id)
                # print(f"Rank {rank + 1}: Document {doc_id + 1}: {documents[doc_id]}")
                # print(self.documents[0])
                self.print_ranked_data(self.documents[doc_id])
                # self.process_documents(self.documents[doc_id])
                # if results:
                #     # ranked_results = self.ranker.rank_results(results)
                #     print("\nSearch Results:")
                #     for doc_id in ranked_results:
                #         print(f"Document {doc_id + 1}: {documents[doc_id]}")
                #
                # else:
                #     print("No results found.")

            ranked_results = self.ranker.rank_vectors_results_reutrn_tuples(similarity_scores)

            # Print top-k ranked documents
            top_k = 10
            for rank, (doc_id, score) in enumerate(ranked_results[:top_k]):
                print(f"Rank {rank + 1}: Document {doc_id + 1}, Similarity Score: {score}")
                self.print_ranked_data(self.documents[doc_id])

                # print(self.documents[doc_id])
                print()

            # return ranked_results

    def run_final(self, query_text):
        # if not self.indexer.document_vectors:
        # self.indexer.index_documents(self.documents)

        # documents = self.indexer.storage_manager.load_processed_docs()
        documents_list = r.read_and_process_file(
            file_path='/home/abdallh/PycharmProjects/information_system/data/processed_docs.txt')

        documents=set(documents_list)
        print(len(documents_list))
        print(len(documents))
        self.documents = r.get_second_values_from_tuples(documents)
        idss = r.get_first_values_from_tuples(documents)
        relevant_idss = []
        # self.documents = self.indexer.storage_manager.load_processed_docs()
        self.indexer.vectorizer = self.indexer.storage_manager.load_vectorizer()
        # self.indexer.document_vectors = self.indexer.storage_manager.load_document_vectors()
        # self.indexer.vectorizer = TfidfVectorizer()
        # self.indexer.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=10000)

        self.indexer.document_vectors = self.indexer.vectorizer.fit_transform(self.documents)
        # # Save the vectorizer and document vectors
        self.indexer.storage_manager.save_vectorizer(self.indexer.vectorizer)
        self.indexer.storage_manager.save_document_vectors(self.indexer.document_vectors)


        while True:
            # query = input("\nEnter your search query (or 'exit' to quit): ")
            # if query.lower() == 'exit':
            #     break
            # Process the query
            # processed_query = self.query_processor.process_query(query)
            processed_query = self.query_processor.complete_process_query(query_text)
            # Join the list of strings into a single string
            joined_string = " ".join(processed_query)
            print(joined_string)
            # Transform the processed query to VSM using the same vectorizer as the documents
            query_vector = self.indexer.vectorizer.transform([joined_string])

            # Perform the search to get similarity scores
            similarity_scores = self.indexer.search_vectors(query_vector)
            #
            ranked_results = self.ranker.rank_vectors_results_reutrn_tuples(similarity_scores)

            #-------------------------------------------------------------------------
            # # Calculate cosine similarity using TF-IDF
            # tfidf_similarity_scores = cosine_similarity(query_vector, self.indexer.document_vectors).flatten()
            # # Initialize BM25
            # bm25 = BM25Okapi(self.documents)
            #
            # # Calculate BM25 scores
            # bm25_scores = bm25.get_scores(joined_string.split())
            #
            # # Combine the scores for ranking (you may experiment with different weighting)
            # combined_scores = (tfidf_similarity_scores + bm25_scores) / 2
            #
            # # Get the indices of documents sorted by combined scores
            # sorted_indices = combined_scores.argsort()[::-1]
            # ranked_results = self.ranker.rank_vectors_results_reutrn_tuples(sorted_indices)
            #------------------------------------------------------------------------

            for rank, (doc_id, score) in enumerate(ranked_results[:10000]):
                relevant_idss.append(idss[doc_id])
            ranked_results_updated = [(idss[doc_id], score) for doc_id, score in ranked_results[:10000]]

            return relevant_idss, ranked_results_updated

    def run_final_dataset2(self, query_text):
        # Load the dataset

        dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")

        # Step 2: Iterate through queries and store them
        queries = {query.query_id: query.text for query in dataset.queries_iter()}
        print("Queries loaded:", queries)

        # Step 3: Extract relevant document IDs from qrels
        qrels = {}
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = []
            qrels[qrel.query_id].append(qrel.doc_id)

        print("Qrels loaded:", qrels)

        # Step 4: Retrieve and read only relevant documents
        # Set to store unique relevant document IDs
        relevant_doc_ids = set()
        for doc_ids in qrels.values():
            relevant_doc_ids.update(doc_ids)

        # Dictionary to store the relevant documents
        relevant_docs = {query_id: [] for query_id in queries.keys()}
        relevant_docs_id = {query_id: [] for query_id in queries.keys()}

        # Iterate through the dataset and collect relevant documents
        for doc in dataset.docs_iter():
            if doc.doc_id in relevant_doc_ids:
                for query_id, doc_ids in qrels.items():
                    if doc.doc_id in doc_ids:
                        relevant_docs[query_id].append(doc)
                        relevant_docs_id[query_id].append(doc.doc_id)

        documents = r.read_and_process_file(
            file_path='/home/abdallh/PycharmProjects/information_system/data/processed_docs_dataset2.txt')
        documents_data = r.get_second_values_from_tuples(documents)
        idss = r.get_first_values_from_tuples(documents)
        relevant_idss = []
        # Collect all document IDs excluding those in exclude_doc_ids
        # all_doc_ids = [doc.doc_id for doc in dataset.docs_iter() if doc.doc_id not in idss]
        # import random
        # sampled_doc_ids = random.sample(all_doc_ids, 5000)
        #
        # # Step 5: Write relevant documents to files
        # # Create output directory if it doesn't exist
        # output_dir = "relevant_docs"
        # os.makedirs(output_dir, exist_ok=True)
        # processed_docs = []
        # for doc in dataset.docs_iter():
        #     if doc.doc_id in sampled_doc_ids:
        #         processed_text = self.query_processor.complete_process_query(
        #             (doc.title.lower() + " ") + doc.summary.lower() + " " + doc.condition + " "
        #         )
        #         processed_text = ' '.join(processed_text)
        #         processed_docs.append((doc.doc_id, processed_text))
        # #
        # # print("Relevant documents saved.")
        # self.indexer.storage_manager.save_processed_docs(processed_docs)
        # return 0, 0, 0

        # documents = self.indexer.storage_manager.load_processed_docs()

        # self.documents = self.indexer.storage_manager.load_processed_docs2()
        # self.indexer.vectorizer = self.indexer.storage_manager.load_vectorizer2()
        # self.indexer.document_vectors = self.indexer.storage_manager.load_document_vectors()
        vectorizer = TfidfVectorizer()
        # self.indexer.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=10000)

        document_vectors = vectorizer.fit_transform(documents_data)
        # # Save the vectorizer and document vectors
        self.indexer.storage_manager.save_vectorizer2(vectorizer)
        self.indexer.storage_manager.save_document_vectors2(document_vectors)

        while True:
            # query = input("\nEnter your search query (or 'exit' to quit): ")
            # if query.lower() == 'exit':
            #     break
            # Process the query
            # processed_query = self.query_processor.process_query(query)
            processed_query = self.query_processor.complete_process_query(query_text)
            # Join the list of strings into a single string
            joined_string = " ".join(processed_query)
            print(joined_string)
            # Transform the processed query to VSM using the same vectorizer as the documents
            query_vector = vectorizer.transform([joined_string])
            # print(type(self.indexer.document_vectors))  # Output: <class 'numpy.ndarray'>
            # print(type(query_vector))

            # Perform the search to get similarity scores
            similarity_scores = self.indexer.search_vectors_ev(query_vector=query_vector,
                                                               document_vectors=document_vectors)

            ranked_results = self.ranker.rank_vectors_results_reutrn_tuples(similarity_scores)
            # print(ranked_results)

            for rank, (doc_id, score) in enumerate(ranked_results[:5000]):
                relevant_idss.append(idss[doc_id])
            ranked_results_updated = [(idss[doc_id], score) for doc_id, score in ranked_results[:10000]]

            return relevant_idss, ranked_results_updated, relevant_docs_id
            # return ranked_results

    def load_dataset(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract documents
        documents = []
        for doc in root.findall('document'):
            doc_id = doc.get('id')
            text = doc.find('text').text
            # Add other metadata as needed
            documents.append((doc_id, text))

        return documents

    def read_jsonl_file(self, file_path):
        documents = []

        # Open the JSON Lines file
        with open(file_path, 'r', encoding='utf-8') as file:
            x = 0

            # Read each line from the file
            for line in file:
                x += 1
                # if x > 100:
                #     break
                print(x)
                # Parse the line as a JSON object
                doc = json.loads(line)
                # Append the document to the list of documents
                documents.append(doc)

        return documents

    def process_documents(self, documents):
        # Process each document as needed for your IR system
        for doc in documents:
            # Extract document fields

            doc_id = doc.get('_id')
            doc_title = doc.get('title')
            doc_text = doc.get('text')
            doc_metadata = doc.get('metadata')

            # Process document data as required by your IR system
            # For example, you might index the document text, title, and metadata

            # Print the document information as a demonstration
            print(f"Document ID: {doc_id}")
            print(f"Title: {doc_title}")
            print(f"Text: {doc_text[:100]}...")  # Print the first 100 characters of the text
            print(f"Metadata: {doc_metadata}")
            print()

    def print_ranked_data(self, doc):
        # Process each document as needed for your IR system
        # Extract document fields
        # print(type(doc))
        # print(doc)
        doc_id = doc.get('_id')
        doc_title = doc.get('title')
        doc_text = doc.get('text')
        doc_metadata = doc.get('metadata')

        # Process document data as required by your IR system
        # For example, you might index the document text, title, and metadata

        # Print the document information as a demonstration
        print(f"Document ID: {doc_id}")
        print(f"Title: {doc_title}")
        print(f"Text: {doc_text[:100]}...")  # Print the first 100 characters of the text
        print(f"Metadata: {doc_metadata}")
        print()

    def print_ranked_dataset_2(self, doc):
        # Process each document as needed for your IR system
        # Extract document fields
        # print(type(doc))
        # print(doc)
        doc_id = doc.doc_id
        doc_title = doc.title
        doc_text = doc.detailed_description
        doc_metadata = doc.summary

        # Process document data as required by your IR system
        # For example, you might index the document text, title, and metadata

        # Print the document information as a demonstration
        print(f"Document ID: {doc_id}")
        print(f"Title: {doc_title}")
        print(f"Text: {doc_text[:100]}...")  # Print the first 100 characters of the text
        print(f"summary: {doc_metadata}")
        print()

    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return lines

    # def get_docs_by_ids(self, file_path, ids):
    #     lines = self.read_file(file_path)
    #     matching_docs = []
    #
    #     for line in lines:
    #         doc = json.loads(line)
    #         if doc['_id'] in ids:
    #             matching_docs.append(doc)
    #
    #     return matching_docs

    def get_docs_by_ids(self, file_path, document_ids):
        matching_documents = {}
        with open(file_path, 'r') as file:
            for line in file:
                doc = json.loads(line)
                doc_id = doc['_id']
                if doc_id in document_ids:
                    matching_documents[doc_id] = doc
                    # If all document IDs have been found, break the loop
                    if len(matching_documents) == len(document_ids):
                        break
        return matching_documents

    def get_docs_by_ids_dataset2(self, file_path, document_ids):
        matching_documents = {}
        dataset = ir_datasets.load(file_path)

        for doc in dataset.docs_iter():
            if doc.doc_id in document_ids:
                matching_documents[doc.doc_id] = doc
                # If all document IDs have been found, break the loop
                if len(matching_documents) == len(document_ids):
                    break
        return matching_documents
