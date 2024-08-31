from itertools import chain

import spacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Barack Obama was the 44th President of the United States. He was born in Hawaii on August 4, 1961."

# Process the text with spaCy
doc = nlp(text)

# Print named entities in the text
print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
import spacy


class QueryProcessor:
    def __init__(self):
        # Load the spaCy language model
        self.nlp = spacy.load("en_core_web_sm")

    def process_query(self, query):
        # Lowercase the query
        query = query.lower()

        # Process the query with spaCy to perform NER
        doc = self.nlp(query)

        # Extract named entities from the query
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]

        # You can use named entities to refine the query or adjust search parameters
        # For now, let's just print the named entities
        print("Named Entities in query:", named_entities)

        # Return the processed query
        return query


from nltk.corpus import wordnet


def expand_query(query):
    words = query.split()
    expanded_query = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            expanded_query.append(synonyms[0].lemmas()[0].name())
        else:
            expanded_query.append(word)
    return ' '.join(expanded_query)


query = 'example query'
expanded_query = expand_query(query)
print('Expanded Query:', expanded_query)

from flask import Flask, request

app = Flask(__name__)


@app.route('/feedback', methods=['POST'])
def feedback():
    user_feedback = request.form.get('feedback')
    # Store the feedback in a database or file
    with open('feedback.txt', 'a') as f:
        f.write(user_feedback + '\n')
    return 'Feedback received', 200


if __name__ == '__main__':
    app.run()

'''that uses Term Frequency-Inverse Document Frequency (TF-IDF) weighting and BM25 to rank documents based on query terms, and natural language processing (NLP) techniques such as Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.
Required Libraries:

    Scikit-learn: For TF-IDF vectorization and cosine similarity calculation.
    RankBM25: For implementing the BM25 retrieval model.
    spaCy: For Named Entity Recognition (NER) and Part-of-Speech (POS) tagging
        TF-IDF weighting: The code uses the TfidfVectorizer from scikit-learn to calculate TF-IDF scores for documents and queries.
    BM25: The code uses the BM25Okapi from the rank-bm25 library to calculate BM25 similarity scores.
    Named Entity Recognition (NER): The code uses spaCy's English model (en_core_web_sm) to identify named entities in the documents and queries.
    Part-of-Speech (POS) Tagging: The code uses spaCy to tag the parts of speech of words in documents and queries.
       The preprocess_text function preprocesses the input text by removing punctuation and converting it to lowercase.
    The analyze_text function uses spaCy to perform Named Entity Recognition (NER) and Part-of-Speech (POS) tagging on the input text.
    The search_ir_system function preprocesses the query and transforms it using the TF-IDF vectorizer. It calculates cosine similarity scores using TF-IDF and BM25 similarity scores.
    The combined scores are calculated as the average of TF-IDF similarity scores and BM25 scores (you can experiment with different weighting).
    The function ranks the documents based on the combined scores and displays the results.
    '''


import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import spacy

# Load spaCy's English model for NER and POS tagging
nlp = spacy.load("en_core_web_sm")

# Sample documents
documents = [
    "I want to eat an apple.",
    "An apple a day keeps the doctor away.",
    "I enjoy eating fresh apples in the morning."
]


# Function to preprocess text (removes punctuation and converts to lowercase)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# Tokenize and preprocess documents
preprocessed_documents = [preprocess_text(doc) for doc in documents]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)

# Initialize BM25
bm25 = BM25Okapi(preprocessed_documents)


# Function to perform NER and POS tagging using spaCy
def analyze_text(text):
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]
    pos_tags = [(token.text, token.pos_) for token in doc]
    return named_entities, pos_tags


# Function to search the IR system using a query
def search_ir_system(query):
    # Preprocess the query
    preprocessed_query = preprocess_text(query)

    # Perform NER and POS tagging on the query
    named_entities, pos_tags = analyze_text(query)
    print(f"Named Entities in query: {named_entities}")
    print(f"POS Tags in query: {pos_tags}")

    # Transform the query using the TF-IDF vectorizer
    query_vector = vectorizer.transform([preprocessed_query])

    # Calculate cosine similarity using TF-IDF
    tfidf_similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Calculate BM25 scores
    bm25_scores = bm25.get_scores(preprocessed_query.split())

    # Combine the scores for ranking (you may experiment with different weighting)
    combined_scores = (tfidf_similarity_scores + bm25_scores) / 2

    # Get the indices of documents sorted by combined scores
    sorted_indices = combined_scores.argsort()[::-1]

    # Display the ranked search results
    print("\nSearch Results (ranked by relevance):")
    for rank, doc_index in enumerate(sorted_indices):
        print(f"Rank {rank + 1}: Document {doc_index + 1}: {documents[doc_index]}")


# Example query
query = "I want an apple"
search_ir_system(query)

'''To implement phrase matching and adjust the retrieval model to handle phrases, you can modify the information retrieval (IR) system to use n-gram tokenization and search for documents that contain the exact phrases. This approach involves considering not just single words (unigrams) but also word pairs (bigrams) and trigrams to preserve the context and order of words in the query.

Here are the key steps to implement phrase matching:

    Adjust the Tokenization: Modify the tokenizer to handle n-grams (phrases) in addition to unigrams. This can be done using the TfidfVectorizer from scikit-learn with an ngram_range parameter.

    Search for Exact Phrases: After tokenization, you can match the query as a phrase or phrases against the documents.

    Combine Scores for Ranking: Calculate similarity scores using cosine similarity, BM25, or other retrieval models. Combine the scores to rank the documents.

    Display the Top Results: Display the top-ranked documents based on their similarity to the query.
        CountVectorizer: Handles n-grams (phrases) in addition to unigrams, using an n-gram range of (1, 2).
    Phrase Queries: query_vector and document_vectors represent the query and documents in the vector space model, considering unigrams and bigrams.
    BM25: The BM25Okapi instance calculates BM25 scores for the query.
    Search Results: The function search_ir_system combines the similarity scores from cosine similarity and BM25, sorts them by relevance, and displays the top-ranked results.

You can adjust the ngram_range in CountVectorizer based on your requirements to consider larger phrases (e.g., trigrams) if necessary.
    '''
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import spacy

# Load spaCy's English model for NER and POS tagging
nlp = spacy.load("en_core_web_sm")

# Sample documents
documents = [
    "I want to eat an apple.",
    "An apple a day keeps the doctor away.",
    "I enjoy eating fresh apples in the morning."
]


# Function to preprocess text (removes punctuation and converts to lowercase)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# Tokenize and preprocess documents
preprocessed_documents = [preprocess_text(doc) for doc in documents]

# Create a CountVectorizer with n-gram range for phrase matching
vectorizer = CountVectorizer(ngram_range=(1, 2))

# Fit the vectorizer on the preprocessed documents and transform them
document_vectors = vectorizer.fit_transform(preprocessed_documents)

# Initialize BM25
bm25 = BM25Okapi(preprocessed_documents)


# Function to perform NER and POS tagging using spaCy
def analyze_text(text):
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]
    pos_tags = [(token.text, token.pos_) for token in doc]
    return named_entities, pos_tags


# Function to search the IR system using a query
def search_ir_system(query):
    # Preprocess the query
    preprocessed_query = preprocess_text(query)

    # Perform NER and POS tagging on the query
    named_entities, pos_tags = analyze_text(query)
    print(f"Named Entities in query: {named_entities}")
    print(f"POS Tags in query: {pos_tags}")

    # Transform the query using the vectorizer (handles n-grams)
    query_vector = vectorizer.transform([preprocessed_query])

    # Calculate cosine similarity using document vectors
    cosine_similarity_scores = cosine_similarity(query_vector, document_vectors).flatten()

    # Calculate BM25 scores
    bm25_scores = bm25.get_scores(preprocessed_query.split())

    # Combine the scores for ranking (you may experiment with different weighting)
    combined_scores = (cosine_similarity_scores + bm25_scores) / 2

    # Get the indices of documents sorted by combined scores
    sorted_indices = combined_scores.argsort()[::-1]

    # Display the ranked search results
    print("\nSearch Results (ranked by relevance):")
    for rank, doc_index in enumerate(sorted_indices):
        print(f"Rank {rank + 1}: Document {doc_index + 1}: {documents[doc_index]}")


# Example query
query = "I want an apple"
search_ir_system(query)


def dynamic_ranking(similarity_scores, user_feedback):
    # Adjust ranking based on user feedback or contextual factors
    if user_feedback == 'positive':
        return similarity_scores * 1.5  # Boost relevance scores
    elif user_feedback == 'negative':
        return similarity_scores * 0.5  # Penalize relevance scores
    else:
        return similarity_scores  # No adjustment

# Apply dynamic ranking strategy
adjusted_scores = dynamic_ranking(similarity_scores, user_feedback)

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer with sparse matrix representation
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', use_idf=True, smooth_idf=True)

# Vectorize documents using sparse matrix representation
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

from nltk.corpus import wordnet

def expand_query(query):
    expanded_query = []
    for word in query.split():
        synonyms = [synset.lemmas()[0].name() for synset in wordnet.synsets(word)]
        expanded_query.extend(synonyms)
    return ' '.join(expanded_query)

expanded_query = expand_query(query)
from nltk.corpus import wordnet

def expand_query(query):
    expanded_query = []
    for word in query.split():
        synonyms = [synset.lemmas()[0].name() for synset in wordnet.synsets(word)]
        expanded_query.extend(synonyms)
    return ' '.join(expanded_query)

expanded_query = expand_query(query)
vector_cache = {}

def vectorize_document(document):
    if document in vector_cache:
        return vector_cache[document]
    else:
        vector = vectorizer.transform([document])
        vector_cache[document] = vector
        return vector

document_vector = vectorize_document(document)

from nltk.corpus import wordnet

def suggest_query_refinement(query):
    synonyms = set()
    for word in query.split():
        for synset in wordnet.synsets(word):
            synonyms.update(synset.lemma_names())
    return list(synonyms)

import gensim.downloader as api

# Load pre-trained Word2Vec model
word2vec_model = api.load('word2vec-google-news-300')

import json
import random
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


class DocumentProcessor:
    def __init__(self, corpus_file, qrels_file, queries_file):
        self.tfidf_matrix = None
        self.corpus_file = corpus_file
        self.qrels_file = qrels_file
        self.queries_file = queries_file
        self.documents = self.load_corpus()
        self.qrels = self.load_qrels()
        self.queries = self.load_queries()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.bm25 = None

    def load_corpus(self):
        with open(self.corpus_file, 'r') as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines]

    def load_qrels(self):
        qrels = pd.read_csv(self.qrels_file, sep='\t', names=["query_id", "corpus_id", "score"])
        return qrels

    def load_queries(self):
        with open(self.queries_file, 'r') as file:
            queries = {json.loads(line)['_id']: json.loads(line)['text'] for line in file}
        return queries

    def process_documents(self):
        docs = [doc['text'] for doc in self.documents]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(docs)
        tokenized_corpus = [doc.split(" ") for doc in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search_query(self, query_text):
        query_vec = self.tfidf_vectorizer.transform([query_text])
        bm25_scores = self.bm25.get_scores(query_text.split())
        tfidf_scores = self.tfidf_matrix.dot(query_vec.T).toarray().flatten()

        combined_scores = 0.5 * tfidf_scores + 0.5 * bm25_scores
        doc_indices = combined_scores.argsort()[::-1][:10]

        return [(self.documents[i]['_id'], combined_scores[i]) for i in doc_indices]

    def evaluate(self):
        all_precision = []
        all_recall = []
        all_f1 = []

        for query_id, query_text in self.queries.items():
            relevant_docs = self.qrels[self.qrels['query_id'] == query_id]['corpus_id'].tolist()
            retrieved_docs = [doc_id for doc_id, score in self.search_query(query_text)]

            evaluator = Evaluate(actual=relevant_docs, predicted=retrieved_docs, k=10)
            metrics = evaluator.get_metrics()

            all_precision.append(metrics['precision_at_k'])
            all_recall.append(metrics['recall'])
            all_f1.append(metrics['F1'])

        average_precision = sum(all_precision) / len(all_precision) if all_precision else 0
        average_recall = sum(all_recall) / len(all_recall) if all_recall else 0
        average_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0

        print("Overall Evaluation Metrics:")
        print("Average Precision:", average_precision)
        print("Average Recall:", average_recall)
        print("Average F1 Score:", average_f1)


def main():
    processor = DocumentProcessor(
        corpus_file='/path/to/corpus.jsonl',
        qrels_file='/path/to/qrels.tsv',
        queries_file='/path/to/queries.jsonl'
    )
    processor.process_documents()
    processor.evaluate()


if __name__ == '__main__':
    main()

import json
import random
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from nltk.corpus import wordnet
from textblob import TextBlob


class DocumentProcessor:
    def __init__(self, corpus_file, qrels_file, queries_file):
        self.corpus_file = corpus_file
        self.qrels_file = qrels_file
        self.queries_file = queries_file
        self.documents = self.load_corpus()
        self.qrels = self.load_qrels()
        self.queries = self.load_queries()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.bm25 = None

    def load_corpus(self):
        with open(self.corpus_file, 'r') as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines]

    def load_qrels(self):
        qrels = pd.read_csv(self.qrels_file, sep='\t', names=["query_id", "corpus_id", "score"])
        return qrels

    def load_queries(self):
        with open(self.queries_file, 'r') as file:
            queries = {json.loads(line)['_id']: json.loads(line)['text'] for line in file}
        return queries

    def process_documents(self):
        docs = [doc['text'] for doc in self.documents]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(docs)
        tokenized_corpus = [doc.split(" ") for doc in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def expand_query(self, query):
        expanded_query = query
        for word in query.split():
            synonyms = wordnet.synsets(word)
            lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
            if lemmas:
                expanded_query += ' ' + ' '.join(lemmas)
        return expanded_query

    def spell_correction(self, query):
        corrected_query = str(TextBlob(query).correct())
        return corrected_query

    def refine_query(self, query):
        corrected_query = self.spell_correction(query)
        expanded_query = self.expand_query(corrected_query)
        return expanded_query

    def search_query(self, query_text):
        refined_query = self.refine_query(query_text)
        query_vec = self.tfidf_vectorizer.transform([refined_query])
        bm25_scores = self.bm25.get_scores(refined_query.split())
        tfidf_scores = self.tfidf_matrix.dot(query_vec.T).toarray().flatten()

        combined_scores = 0.5 * tfidf_scores + 0.5 * bm25_scores
        doc_indices = combined_scores.argsort()[::-1][:10]

        return [(self.documents[i]['_id'], combined_scores[i]) for i in doc_indices]

    def evaluate(self):
        all_precision = []
        all_recall = []
        all_f1 = []

        for query_id, query_text in self.queries.items():
            relevant_docs = self.qrels[self.qrels['query_id'] == query_id]['corpus_id'].tolist()
            retrieved_docs = [doc_id for doc_id, score in self.search_query(query_text)]

            evaluator = Evaluate(actual=relevant_docs, predicted=retrieved_docs, k=10)
            metrics = evaluator.get_metrics()

            all_precision.append(metrics['precision_at_k'])
            all_recall.append(metrics['recall'])
            all_f1.append(metrics['F1'])

        average_precision = sum(all_precision) / len(all_precision) if all_precision else 0
        average_recall = sum(all_recall) / len(all_recall) if all_recall else 0
        average_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0

        print("Overall Evaluation Metrics:")
        print("Average Precision:", average_precision)
        print("Average Recall:", average_recall)
        print("Average F1 Score:", average_f1)


def main():
    processor = DocumentProcessor(
        corpus_file='/home/abdallh/Documents/webis-touche2020/corpus.jsonl',
        qrels_file='/home/abdallh/Documents/webis-touche2020/qrels.tsv',
        queries_file='/home/abdallh/Documents/webis-touche2020/queries.jsonl'
    )
    processor.process_documents()
    processor.evaluate()


if __name__ == '__main__':
    main()
'''Explanation

    Expand Query:
        Uses wordnet to find synonyms and expand the query.
        Adds synonyms to the original query to broaden the search.

    Spell Correction:
        Uses textblob to correct spelling mistakes in the query.
        Ensures the query is free from spelling errors.

    Refine Query:
        Combines spell correction and query expansion to refine the query.
        Provides a refined query for better search results.

    Search Query:
        Searches using the refined query.
        Computes combined scores from TF-IDF and BM25.

    Evaluate:
        Evaluates the search results for all queries.
        Calculates average precision, recall, and F1 score.

By integrating query refinement features into your system, 
you can improve the relevance of search results and provide
 better assistance to users in formulating their queries.'''



import json
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

class DocumentProcessor:
    def __init__(self, corpus_file, qrels_file, queries_file):
        self.document_embeddings = None
        self.corpus_file = corpus_file
        self.qrels_file = qrels_file
        self.queries_file = queries_file
        self.documents = self.load_corpus()
        self.qrels = self.load_qrels()
        self.queries = self.load_queries()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_corpus(self):
        with open(self.corpus_file, 'r') as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines]

    def load_qrels(self):
        qrels = pd.read_csv(self.qrels_file, sep='\t', names=["query_id", "corpus_id", "score"])
        return qrels

    def load_queries(self):
        with open(self.queries_file, 'r') as file:
            queries = {json.loads(line)['_id']: json.loads(line)['text'] for line in file}
        return queries

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def process_documents(self):
        self.document_embeddings = []
        for doc in self.documents:
            text = doc['title'] + " " + doc['text']
            embedding = self.encode_text(text)
            self.document_embeddings.append((doc['_id'], embedding))
        self.document_embeddings = {doc_id: emb for doc_id, emb in self.document_embeddings}

    def search_query(self, query_text):
        query_embedding = self.encode_text(query_text)
        similarities = {}
        for doc_id, doc_embedding in self.document_embeddings.items():
            similarities[doc_id] = cosine_similarity(query_embedding, doc_embedding).flatten()[0]

        sorted_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:10]
        return sorted_docs

    def evaluate(self):
        all_precision = []
        all_recall = []
        all_f1 = []

        for query_id, query_text in self.queries.items():
            relevant_docs = self.qrels[self.qrels['query_id'] == query_id]['corpus_id'].tolist()
            retrieved_docs = [doc_id for doc_id, score in self.search_query(query_text)]

            evaluator = Evaluate(actual=relevant_docs, predicted=retrieved_docs, k=10)
            metrics = evaluator.get_metrics()

            all_precision.append(metrics['precision_at_k'])
            all_recall.append(metrics['recall'])
            all_f1.append(metrics['F1'])

        average_precision = sum(all_precision) / len(all_precision) if all_precision else 0
        average_recall = sum(all_recall) / len(all_recall) if all_recall else 0
        average_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0

        print("Overall Evaluation Metrics:")
        print("Average Precision:", average_precision)
        print("Average Recall:", average_recall)
        print("Average F1 Score:", average_f1)

class Evaluate:
    def __init__(self, actual, predicted, k):
        self.actual = actual
        self.predicted = predicted
        self.k = k

    def precision_at_k(self, actual, predicted, k):
        predicted = predicted[:k]
        tp = len(set(predicted) & set(actual))
        return tp / k

    def recall(self, actual, predicted):
        tp = len(set(predicted) & set(actual))
        return tp / len(actual)

    def average_precision_at_k(self, actual, predicted, k):
        precisions = [self.precision_at_k(actual, predicted, i + 1) for i in range(k) if predicted[i] in actual]
        if not precisions:
            return 0
        return sum(precisions) / min(k, len(actual))

    def mean_reciprocal_rank(self, actual, predicted):
        for i, p in enumerate(predicted):
            if p in actual:
                return 1 / (i + 1)
        return 0

    def get_metrics(self):
        precision = self.precision_at_k(self.actual, self.predicted, self.k)
        recall = self.recall(self.actual, self.predicted)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        map_k = self.average_precision_at_k(self.actual, self.predicted, self.k)
        mrr = self.mean_reciprocal_rank(self.actual, self.predicted)

        return {
            'precision_at_k': precision,
            'recall': recall,
            'F1': f1,
            'MAP': map_k,
            'MRR': mrr
        }

    def print_all(self):
        metrics = self.get_metrics()
        print(f"Precision@{self.k}: {metrics['precision_at_k']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1: {metrics['F1']}")
        print(f"MAP: {metrics['MAP']}")
        print(f"MRR: {metrics['MRR']}")

def main():
    processor = DocumentProcessor(
        corpus_file='/home/abdallh/Documents/webis-touche2020/corpus.jsonl',
        qrels_file='/home/abdallh/Documents/webis-touche2020/qrels.tsv',
        queries_file='/home/abdallh/Documents/webis-touche2020/queries.jsonl'
    )
    processor.process_documents()
    processor.evaluate()

if __name__ == '__main__':
    main()




import json
import pandas as pd
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

class MultilingualDocumentProcessor:
    def __init__(self, corpus_file, qrels_file, queries_file):
        self.document_embeddings = None
        self.corpus_file = corpus_file
        self.qrels_file = qrels_file
        self.queries_file = queries_file
        self.documents = self.load_corpus()
        self.qrels = self.load_qrels()
        self.queries = self.load_queries()
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_corpus(self):
        with open(self.corpus_file, 'r') as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines]

    def load_qrels(self):
        qrels = pd.read_csv(self.qrels_file, sep='\t', names=["query_id", "corpus_id", "score"])
        return qrels

    def load_queries(self):
        with open(self.queries_file, 'r') as file:
            queries = {json.loads(line)['_id']: json.loads(line)['text'] for line in file}
        return queries

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def process_documents(self):
        self.document_embeddings = []
        for doc in self.documents:
            text = doc['title'] + " " + doc['text']
            embedding = self.encode_text(text)
            self.document_embeddings.append((doc['_id'], embedding))
        self.document_embeddings = {doc_id: emb for doc_id, emb in self.document_embeddings}

    def search_query(self, query_text):
        query_embedding = self.encode_text(query_text)
        similarities = {}
        for doc_id, doc_embedding in self.document_embeddings.items():
            similarities[doc_id] = cosine_similarity(query_embedding, doc_embedding).flatten()[0]

        sorted_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:10]
        return sorted_docs

    def evaluate(self):
        all_precision = []
        all_recall = []
        all_f1 = []

        for query_id, query_text in self.queries.items():
            relevant_docs = self.qrels[self.qrels['query_id'] == query_id]['corpus_id'].tolist()
            retrieved_docs = [doc_id for doc_id, score in self.search_query(query_text)]

            evaluator = Evaluate(actual=relevant_docs, predicted=retrieved_docs, k=10)
            metrics = evaluator.get_metrics()

            all_precision.append(metrics['precision_at_k'])
            all_recall.append(metrics['recall'])
            all_f1.append(metrics['F1'])

        average_precision = sum(all_precision) / len(all_precision) if all_precision else 0
        average_recall = sum(all_recall) / len(all_recall) if all_recall else 0
        average_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0

        print("Overall Evaluation Metrics:")
        print("Average Precision:", average_precision)
        print("Average Recall:", average_recall)
        print("Average F1 Score:", average_f1)

class Evaluate:
    def __init__(self, actual, predicted, k):
        self.actual = actual
        self.predicted = predicted
        self.k = k

    def precision_at_k(self, actual, predicted, k):
        predicted = predicted[:k]
        tp = len(set(predicted) & set(actual))
        return tp / k

    def recall(self, actual, predicted):
        tp = len(set(predicted) & set(actual))
        return tp / len(actual)

    def average_precision_at_k(self, actual, predicted, k):
        precisions = [self.precision_at_k(actual, predicted, i + 1) for i in range(k) if predicted[i] in actual]
        if not precisions:
            return 0
        return sum(precisions) / min(k, len(actual))

    def mean_reciprocal_rank(self, actual, predicted):
        for i, p in enumerate(predicted):
            if p in actual:
                return 1 / (i + 1)
        return 0

    def get_metrics(self):
        precision = self.precision_at_k(self.actual, self.predicted, self.k)
        recall = self.recall(self.actual, self.predicted)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        map_k = self.average_precision_at_k(self.actual, self.predicted, self.k)
        mrr = self.mean_reciprocal_rank(self.actual, self.predicted)

        return {
            'precision_at_k': precision,
            'recall': recall,
            'F1': f1,
            'MAP': map_k,
            'MRR': mrr
        }

    def print_all(self):
        metrics = self.get_metrics()
        print(f"Precision@{self.k}: {metrics['precision_at_k']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1: {metrics['F1']}")
        print(f"MAP: {metrics['MAP']}")
        print(f"MRR: {metrics['MRR']}")

def main():
    processor = MultilingualDocumentProcessor(
        corpus_file='/home/abdallh/Documents/webis-touche2020/corpus.jsonl',
        qrels_file='/home/abdallh/Documents/webis-touche2020/qrels.tsv',
        queries_file='/home/abdallh/Documents/webis-touche2020/queries.jsonl'
    )
    processor.process_documents()
    processor.evaluate()

if __name__ == '__main__':
    main()
