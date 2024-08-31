import json
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
'''    Loading Pre-trained BERT:
        Initialize the BERT tokenizer and model.
        Move the model to the appropriate device (CPU/GPU).

    Encoding Text with BERT:
        Encode both documents and queries into embeddings using BERT.
        Use the mean of the token embeddings as the document/query embedding.

    Cosine Similarity:
        Compute the cosine similarity between the query embedding and document embeddings.
        Sort the documents based on similarity scores.

    Evaluation:
        Evaluate the retrieval results using precision, recall, F1, MAP, and MRR.'''
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
        return [json.loads(line) for line in lines][:100]

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

