# import json
# import random
# import os
# import sys
# import pandas as pd
#
from apis.api import query_processor, storage_manager
#
#
# def process_batch(documents):
#     document_ids = [doc['_id'] for doc in documents]
#     processed_docs = []
#     # Weight factor for the titles
#     title_weight = 4
#     # self.processed_docs
#     for doc in documents:
#         doc_id = doc.get('_id')  # Assuming '_id' is the key for document ID
#         processed_text = query_processor.complete_process_query(
#             title_weight * (doc.get('title', ' ').lower() + " ") + doc.get('text', '').lower()
#         )
#
#         processed_text = ' '.join(processed_text)
#         processed_docs.append((doc_id, processed_text))
#
#     processed_documents = []
#     for doc_id, processed_text in processed_docs:
#         processed_documents.append(processed_text)
#
#     storage_manager.save_processed_docs(processed_docs)
#     processed_docs.clear()
#     # self.processed_docs.clear()
#     # self.storage_manager.save_batch_document_vectors(batch_vectors, batch_id)
#
#
# def load_corpus(file_path):
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#     data = [json.loads(line) for line in lines]
#     df = pd.DataFrame(data)
#     return df
#
#
# def get_relevant_and_random_docs(df, relevant_doc_ids, num_random_records=200000):
#     # Filter relevant documents
#     relevant_docs_df = df[df['_id'].isin(relevant_doc_ids)]
#
#     # # Get the remaining documents excluding the relevant ones
#     # remaining_docs_df = df[~df['_id'].isin(relevant_doc_ids)]
#     #
#     # # Sample random records from the remaining documents
#     # random_docs_df = remaining_docs_df.sample(n=num_random_records, random_state=42)
#     #
#     # # Combine relevant and random documents
#     # combined_docs_df = pd.concat([relevant_docs_df, random_docs_df])
#
#     return relevant_docs_df
#
#
# def glue(ids):
#     data = load_corpus(file_path='/home/abdallh/Documents/webis-touche2020/corpus.jsonl')
#     docs = get_relevant_and_random_docs(df=data, relevant_doc_ids=ids)
#     process_batch(documents=docs)
# import json
# import os
# import sys
#
# from apis.api import query_processor, storage_manager
#
#
# def process_batch(documents):
#     processed_docs = []
#     # Weight factor for the titles
#     title_weight = 4
#     for doc in documents:
#         doc_id = doc.get('_id')  # Assuming '_id' is the key for document ID
#         processed_text = query_processor.complete_process_query(
#             title_weight * (doc.get('title', ' ').lower() + " ") + doc.get('text', '').lower()
#         )
#
#         processed_text = ' '.join(processed_text)
#         processed_docs.append((doc_id, processed_text))
#
#     storage_manager.save_processed_docs(processed_docs)
#
#
def load_relevant_doc_ids(file_path):
    relevant_doc_ids = set()
    with open(file_path, 'r') as file:
        for line in file:
            query_id, corpus_id, _ = line.split()
            relevant_doc_ids.add(corpus_id)
    return relevant_doc_ids
#
#
# def get_relevant_docs(corpus_file, relevant_doc_ids):
#     relevant_docs = []
#
#     with open(corpus_file, 'r') as file:
#         for line in file:
#             doc = json.loads(line)
#             doc_id = doc['_id']
#             if doc_id in relevant_doc_ids:
#                 relevant_docs.append(doc)
#
#     return relevant_docs
#
#
# def process_documents(file_path, relevant_doc_ids, batch_size=1000):
#     documents_to_process = get_relevant_docs(file_path, relevant_doc_ids)
#
#     for i in range(0, len(documents_to_process), batch_size):
#         batch = documents_to_process[i:i + batch_size]
#         process_batch(batch)
#
#     print("Processing complete.")
#
#
# def glue(ids):
#     # Assuming the corpus file path and qrels file path are given correctly
#     corpus_file_path = '/home/abdallh/Documents/webis-touche2020/corpus.jsonl'
#     relevant_doc_ids = load_relevant_doc_ids(file_path='/home/abdallh/Documents/webis-touche2020/qrels/test.tsv')
#     print(len(relevant_doc_ids))
#
#     process_documents(corpus_file_path, relevant_doc_ids)
#
#
# # Example usage
# relevant_ids = ["7f546086-2019-04-18T16:57:49Z-00004-000", "7f546086-2019-04-18T16:57:49Z-00005-000"]
# glue(relevant_ids)

import random


def load_tuples(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [eval(line.strip()) for line in lines]


def save_tuples(tuples, file_path):
    with open(file_path, 'w') as file:
        for item in tuples:
            file.write(f"{item}\n")


def delete_random_documents(file_path, keep_ids, num_to_delete):
    # Load all tuples from the file
    tuples = load_tuples(file_path)

    # Filter out tuples whose IDs are not in the keep_ids list
    filtered_tuples = [t for t in tuples if t[0] in keep_ids]

    # Separate tuples to potentially delete
    deletable_tuples = [t for t in tuples if t[0] not in keep_ids]

    # Randomly select tuples to delete
    tuples_to_delete = random.sample(deletable_tuples, min(num_to_delete, len(deletable_tuples)))

    # Create the final list of tuples to keep
    remaining_tuples = filtered_tuples + [t for t in deletable_tuples if t not in tuples_to_delete]

    # Save the remaining tuples back to the file
    save_tuples(remaining_tuples, file_path)

relevant_doc_ids = load_relevant_doc_ids(file_path='/home/abdallh/Documents/webis-touche2020/qrels/test.tsv')
print(len(relevant_doc_ids))
# Example usage
keep_ids = relevant_doc_ids
# example list of IDs to keep
file_path = '/home/abdallh/PycharmProjects/information_system/data/processed_docs.txt'
num_to_delete = 60000

delete_random_documents(file_path, keep_ids, num_to_delete)
