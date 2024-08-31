from interface.cli_interface import CLIInterface
# import nltk
# from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer


# /home/abdallh/Documents/webis-touche2020/

def main():
    interface = CLIInterface()
    interface.load_data('/home/abdallh/Documents/webis-touche2020/corpus.jsonl')
    qrels = interface.load_qrels('/home/abdallh/Documents/webis-touche2020/qrels/test.tsv')
    interface.load_queries('/home/abdallh/Documents/webis-touche2020/queries.jsonl')
    interface.run()
    # Example of Searching and Evaluating
    # query_id = '35'
    # query_text = interface.queries[query_id]
    # retrieved_docs = interface.search2(query_text)
    # precision, recall = interface.evaluate(query_id, retrieved_docs, qrels)
    #
    # print(f"Precision: {precision}, Recall: {recall}")

if __name__ == '__main__':
    main()
    # preprocessed_documents = [
    #     ['Apple', 'fruit', 'i'],
    #     ['Banana', 'also', 'fruit', 'i'],
    #     ['Both', 'apple', 'banana', 'healthy', 'i'],
    #     ['Orange', 'another', 'type', 'fruit', 'i']
    # ]
    # processed_documents = [' '.join(doc) for doc in preprocessed_documents]
    # print(processed_documents)

    # Sample data
    # documents = [
    #     "Apple is a fruit.",
    #     "Banana is also a fruit.",
    #     "Both apple and banana are healthy.",
    #     "Orange is another type of fruit."
    # ]
    #
    # query = "I like apple and banana."
    #
    # # Create a TF-IDF vectorizer
    # vectorizer = TfidfVectorizer()
    #
    # # Fit the vectorizer to the documents and transform the documents to VSM
    # document_vectors = vectorizer.fit_transform(documents)
    #
    # # Transform the query to VSM using the same vectorizer
    # query_vector = vectorizer.transform([query])
    #
    # # Display the vocabulary (terms in the vector space)
    # vocabulary = vectorizer.get_feature_names_out()
    # print("Vocabulary (terms):", vocabulary)
    #
    # # Display the TF-IDF values for each document
    # print("\nTF-IDF values for each document:")
    # for doc_id, doc_vector in enumerate(document_vectors):
    #     print(f"Document {doc_id + 1}:")
    #     tfidf_values = doc_vector.toarray().flatten()  # Convert the sparse matrix to a 1D array
    #     print(dict(zip(vocabulary, tfidf_values)))
    #
    # # Display the TF-IDF values for the query
    # print("\nTF-IDF values for the query:")
    # query_tfidf_values = query_vector.toarray().flatten()  # Convert the sparse matrix to a 1D array
    # print(dict(zip(vocabulary, query_tfidf_values)))

    # nltk.download()
