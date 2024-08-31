# Example usage
# actual = [1, 2, 3]
# predicted = [3, 1, 5, 2, 7]
# k = 3


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score


# from your_ir_system import InformationRetrievalSystem  # Import your IR system
#
# # Assuming you have your dataset in a list of documents and their corresponding relevance labels
# documents = [...]
# relevance_labels = [...]
#
# # Split the data into training, validation, and test sets
# train_docs, temp_docs, train_labels, temp_labels = train_test_split(documents, relevance_labels, test_size=0.4)
# validation_docs, test_docs, validation_labels, test_labels = train_test_split(temp_docs, temp_labels, test_size=0.5)
#
# # Initialize your IR system and train it on the training set
# ir_system = InformationRetrievalSystem()
# ir_system.train(train_docs, train_labels)
#
# # Evaluate the IR system on the validation set
# validation_results = ir_system.search(validation_docs)
# precision = precision_score(validation_labels, validation_results)
# recall = recall_score(validation_labels, validation_results)
# f1 = f1_score(validation_labels, validation_results)
# ndcg = ndcg_score(validation_labels, validation_results)
#
# print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1}, NDCG: {ndcg}')
#
# # Adjust model parameters based on validation results and iterate as needed
# # ...
#
# # After tuning, evaluate on the test set
# test_results = ir_system.search(test_docs)
# test_precision = precision_score(test_labels, test_results)
# test_recall = recall_score(test_labels, test_results)
# test_f1 = f1_score(test_labels, test_results)
# test_ndcg = ndcg_score(test_labels, test_results)
#
# print(f'Test Precision: {test_precision}, Test Recall: {test_recall}, Test F1-Score: {test_f1}, Test NDCG: {test_ndcg}')


# 1. Precision@10: Proportion of relevant documents among the top 10 retrieved documents.
# 2. Recall: Proportion of relevant documents that were retrieved out of the total number of relevant documents.
# 3. MAP (Mean Average Precision): Average of precision values at different recall levels.
# 4. Precision: Proportion of retrieved documents that are relevant.
# 5. MRR (Mean Reciprocal Rank): Average of the reciprocal ranks of the first relevant document found.
# 6. Rank Reciprocal Mean: Average reciprocal rank for all queries.
# Balance Precision and Recall: Depending on your use case,
# decide whether to prioritize precision (reducing false positives) or
# recall (reducing false negatives).
# Consider NDCG and MAP: For evaluating ranked results,
# NDCG and MAP can provide insights into how well your system
# ranks the most relevant documents.
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

    def f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    # def print_all(self):
    #     print("Precision@{}: {}".format(self.k, self.precision_at_k(self.actual, self.predicted, self.k)))
    #     print("Recall: {}".format(self.recall(self.actual, self.predicted)))
    #     print("MAP: {}".format(self.average_precision_at_k(self.actual, self.predicted, self.k)))
    #     print("MRR: {}".format(self.mean_reciprocal_rank(self.actual, self.predicted)))
    def get_metrics(self):
        metrics = {
            'precision_at_k': self.precision_at_k(self.actual, self.predicted, self.k),
            'recall': self.recall(self.actual, self.predicted),
            'MAP': self.average_precision_at_k(self.actual, self.predicted, self.k),
            'MRR': self.mean_reciprocal_rank(self.actual, self.predicted)
        }
        return metrics

    def calculate_metrics(self):
        precision = self.precision_at_k(self.actual, self.predicted, self.k)
        recall = self.recall(self.actual, self.predicted)
        f1 = self.f1_score(precision, recall)
        return precision, recall, f1

    def print_all(self):
        precision = self.precision_at_k(self.actual, self.predicted, self.k)
        recall = self.recall(self.actual, self.predicted)
        f1 = self.f1_score(precision, recall)
        print("Precision@{}: {}".format(self.k, precision))
        print("Recall: {}".format(recall))
        print("F1 Score: {}".format(f1))
        print("MAP: {}".format(self.average_precision_at_k(self.actual, self.predicted, self.k)))
        print("MRR: {}".format(self.mean_reciprocal_rank(self.actual, self.predicted)))
