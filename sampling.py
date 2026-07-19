import torch

def sample_next_token(
    logits: torch.Tensor,
    method: str = "greedy",
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None
) -> torch.Tensor:
    """Samples the next token from logits according to the chosen decoding strategy.
    
    Args:
        logits (torch.Tensor): Logits of shape (batch_size, vocab_size) or (vocab_size,)
        method (str): Sampling method ('greedy', 'temperature', 'top_k', 'top_p')
        temperature (float): Scaling factor (higher means more random, lower means more deterministic)
        top_k (int): Keep only the top k highest probability tokens
        top_p (float): Keep only the top cumulative probability tokens (nucleus sampling)
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # Handle greedy directly
    if method == "greedy":
        return torch.argmax(logits, dim=-1)

    # Scale by temperature
    if temperature > 0.0:
        logits = logits / temperature

    # Apply Top-K filtering
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        min_values = values[:, [-1]]
        logits = torch.where(logits < min_values, torch.tensor(float('-inf'), device=logits.device), logits)

    # Apply Top-P (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p (but keep at least the top 1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Map back to original logit indices
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = torch.where(indices_to_remove, torch.tensor(float('-inf'), device=logits.device), logits)

    # Sample from the filtered distribution
    probs = torch.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens.squeeze(-1)
