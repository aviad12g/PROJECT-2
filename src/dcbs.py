import torch
from abc import ABC, abstractmethod

def deterministic_category_sampling(logits, cluster_ids):
    """
    Greedy cluster, then greedy token.
    Returns `int` → next-token id.
    """
    probs = torch.softmax(logits, dim=-1)
    cluster_mass = torch.zeros_like(torch.bincount(cluster_ids))
    cluster_mass.scatter_add_(0, cluster_ids, probs)
    best_cluster = torch.argmax(cluster_mass)
    token_mask = cluster_ids.eq(best_cluster)
    return torch.argmax(probs.masked_fill(~token_mask, -1e9)).item()

class SamplingStrategy(ABC):
    """
    Abstract base class for sampling strategies.
    """
    @abstractmethod
    def select(self, logits: torch.Tensor) -> int:
        """
        Select next token given logits.
        Returns `int` → next-token id.
        """
        pass

class DeterministicCategoryStrategy(SamplingStrategy):
    """
    Deterministic Category-Based Sampling (DCBS) strategy.
    Implements greedy cluster, then greedy token sampling.
    """
    def __init__(self, cluster_ids: torch.Tensor):
        """
        Initialize the strategy with pre-computed cluster_ids.
        
        Args:
            cluster_ids: Tensor of shape (vocab_size,) containing cluster IDs for each token
        """
        self.cluster_ids = cluster_ids.to(torch.long)
    
    def select(self, logits):
        """
        Select next token using deterministic category-based sampling.
        
        Args:
            logits: Tensor of shape (vocab_size,) containing logits for next token
            
        Returns:
            int: next token ID
        """
        return deterministic_category_sampling(logits, self.cluster_ids)

def generate_with_strategy(model, tokenizer, input_ids, strategy, max_new_tokens):
    generated = []
    for _ in range(max_new_tokens):
        logits = model(input_ids)[0][:, -1, :]
        next_id = strategy.select(logits.squeeze(0))
        generated.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=input_ids.device)], dim=-1)
        if next_id == tokenizer.eos_token_id:
            break
    return torch.tensor(generated, device=model.device) 