import unittest
import torch
from src.dcbs import deterministic_category_sampling, DeterministicCategoryStrategy

class TestDeterministicCategorySampling(unittest.TestCase):
    def test_cluster_selection(self):
        # Create a simple test case with known cluster IDs and logits
        logits = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5, 0.1])
        cluster_ids = torch.tensor([0, 0, 1, 1, 2, 2])
        
        # The expected behavior:
        # Cluster 0 has tokens [0, 1] with probs proportional to [1.0, 2.0] -> sum = 3.0
        # Cluster 1 has tokens [2, 3] with probs proportional to [0.5, 3.0] -> sum = 3.5
        # Cluster 2 has tokens [4, 5] with probs proportional to [1.5, 0.1] -> sum = 1.6
        # Best cluster should be 1, and best token in that cluster should be 3
        
        # Get the result from our sampling function
        result = deterministic_category_sampling(logits, cluster_ids)
        
        # The expected token is 3 (index 3), which has the highest logit in the best cluster
        self.assertEqual(result, 3)
    
    def test_strategy_matches_direct_call(self):
        # Create test case
        logits = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5, 0.1])
        cluster_ids = torch.tensor([0, 0, 1, 1, 2, 2])
        
        # Get result from direct call
        direct_result = deterministic_category_sampling(logits, cluster_ids)
        
        # Get result from strategy class
        strategy = DeterministicCategoryStrategy(cluster_ids)
        strategy_result = strategy.select(logits)
        
        # Results should match
        self.assertEqual(direct_result, strategy_result)
    
    def test_equivalent_to_argmax_when_all_same_cluster(self):
        # When all tokens belong to the same cluster, result should be the same as torch.argmax
        logits = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5, 0.1])
        cluster_ids = torch.tensor([0, 0, 0, 0, 0, 0])  # All tokens in cluster 0
        
        # Get the greedy result
        greedy_result = torch.argmax(logits).item()
        
        # Get the DCBS result
        dcbs_result = deterministic_category_sampling(logits, cluster_ids)
        
        # They should be the same
        self.assertEqual(dcbs_result, greedy_result)

if __name__ == "__main__":
    unittest.main() 