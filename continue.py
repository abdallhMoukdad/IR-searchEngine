import json


def read_processed_ids(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    processed_ids = [json.loads(line.strip())['_id'] for line in lines]
    return processed_ids


def read_corpus_records(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    records = [json.loads(line.strip()) for line in lines]
    return records


def process_remaining_docs(corpus_file, processed_docs_file):
    processed_ids = read_processed_ids(processed_docs_file)
    corpus_records = read_corpus_records(corpus_file)

    remaining_docs = [record for record in corpus_records if record['_id'] not in processed_ids]

    # Process remaining documents here
    for doc in remaining_docs:
        # Process document
        pass


# Example usage
corpus_file_path = '/home/abdallh/Documents/webis-touche2020/corpus.jsonl'

processed_docs_file_path = 'processed_docs.txt'
process_remaining_docs(corpus_file_path, processed_docs_file_path)
