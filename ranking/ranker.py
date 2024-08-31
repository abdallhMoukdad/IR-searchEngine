import numpy as np
class Ranker:
    def __init__(self):
        self.feedback_file = '/home/abdallh/PycharmProjects/information_system/apis/feedback.txt'
        self.user_feedback = self.load_feedback()
    def rank_results(self, results):
        # Basic ranking based on document frequency
        # You can implement more sophisticated ranking algorithms
        # For this example, we'll return results in the order of appearance
        return results

    def load_feedback(self):
        feedback = {}
        with open(self.feedback_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                doc_id = parts[0]
                is_relevant = parts[1] == 'True'
                feedback[doc_id] = is_relevant
        return feedback

    def dynamic_ranking(self, similarity_scores):
        sorted_indices = np.argsort(similarity_scores)[::-1]
        for doc_id in sorted_indices:
            if doc_id in self.user_feedback:
                if self.user_feedback[doc_id]:  # Relevant
                    similarity_scores[doc_id] *= 1.5  # Boost relevance scores
                else:  # Not Relevant
                    similarity_scores[doc_id] *= 0.5  # Penalize relevance scores
        return similarity_scores
    # def dynamic_ranking(self,similarity_scores, user_feedback):
    #     sorted_indices = np.argsort(similarity_scores)[::-1]
    #
    #     # Create a list of tuples (doc_id, score)
    #     ranked_results = [(doc_id, similarity_scores[doc_id]) for doc_id in sorted_indices]
    #     for doc_id in sorted_indices:
    #         if doc_id in user_feedback:
    #             if user_feedback == 'positive':
    #                 return similarity_scores * 1.5  # Boost relevance scores
    #             elif user_feedback == 'negative':
    #                 return similarity_scores * 0.5  # Penalize relevance scores
    #             else:
    #                 return similarity_scores  # No adjustment

        # # Adjust ranking based on user feedback or contextual factors
        # if user_feedback == 'positive':
        #     return similarity_scores * 1.5  # Boost relevance scores
        # elif user_feedback == 'negative':
        #     return similarity_scores * 0.5  # Penalize relevance scores
        # else:
        #     return similarity_scores  # No adjustment

    def rank_vectors_results(self, similarity_scores):
        # Convert similarity scores to an array and sort the indices in descending order
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # Return sorted indices, which indicate document IDs in descending order of relevance
        return sorted_indices

    def rank_vectors_results_reutrn_tuples(self, similarity_scores):
        # Convert similarity scores to an array and sort the indices in descending order
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # Create a list of tuples (doc_id, score)
        ranked_results = [(doc_id, similarity_scores[doc_id]) for doc_id in sorted_indices]

        # Return the list of tuples
        return ranked_results
