from evaluate.evaluate import Evaluate
from interface.cli_interface import CLIInterface


def get_all_relevant_docs(query_ids, qrels):
    """
    Given a list of query IDs and qrels dataframe, return a list of all relevant document IDs.

    :param query_ids: List of query IDs
    :param qrels: DataFrame of qrel lines with columns ['query_id', 'corpus_id', 'score']
    :return: List of all relevant document IDs
    """
    relevant_docs = qrels[qrels['query_id'].isin(query_ids)]['corpus_id'].tolist()
    return relevant_docs


def filter_documents_by_ids(documents, ids):
    return [doc for doc in documents if doc['_id'] in ids]


def main():
    interface = CLIInterface()
    # Load the qrels and queries
    qrels = interface.load_qrels('/home/abdallh/Documents/webis-touche2020/qrels/test.tsv')
    interface.load_queries('/home/abdallh/Documents/webis-touche2020/queries.jsonl')

    # Get all query IDs
    query_ids = list(interface.queries.keys())

    # Initialize metrics
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    query_count = len(query_ids)
    with open('/home/abdallh/Documents/webis-touche2020/qrels/test.tsv', 'r') as f:
        qrels = f.readlines()

    # Iterate over each query
    for query_id in query_ids:
        query_text = interface.queries[query_id]

        # Get relevant documents for the query
        # relevant_docs = [line.split()[1] for line in qrels if line.split()[0] == query_id]
        print(query_id)
        relevant_docs = []
        for line in qrels:
            parts = line.split()
            query_id_str = str(query_id)  # Convert query_id to string for comparison
            # if parts[0] == query_id_str and float(parts[2]) > 0:  # Check if score is greater than zero
            if parts[0] == query_id_str:  # Check if score is greater than zero
                relevant_docs.append(parts[1])
        # print("Relevant Docs:", relevant_docs)

        # print(query_ids)
        # print(len(relevant_docs))
        # Perform the search for the query
        retrieved_ids, retrieved_results = interface.run_final(query_text)
        retrieved_docs_id = retrieved_ids

        # Print top 10 results for debugging
        matching_documents = interface.get_docs_by_ids('/home/abdallh/Documents/webis-touche2020/corpus.jsonl',
                                                       retrieved_ids)
        for rank, (doc_id, score) in enumerate(retrieved_results[:10]):
            document = matching_documents.get(doc_id)
            if document:
                print(f"Rank {rank + 1}: Document {doc_id}, Similarity Score: {score}")
                interface.print_ranked_data(document)
            else:
                print(f"Rank {rank + 1}: Document {doc_id} not found")

        # Evaluate the results
        e = Evaluate(actual=relevant_docs, predicted=retrieved_docs_id, k=10)
        precision, recall, f1 = e.calculate_metrics()

        # Accumulate metrics
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        e.print_all()
        # Print metrics for the current query
        # print(f"Query ID: {query_id}")
        # print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
        print()
        # break

    # Calculate and print overall metrics
    avg_precision = total_precision / query_count
    avg_recall = total_recall / query_count
    avg_f1 = total_f1 / query_count

    print(f"Overall Evaluation Metrics:")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1 Score: {avg_f1}")


if __name__ == '__main__':
    main()
