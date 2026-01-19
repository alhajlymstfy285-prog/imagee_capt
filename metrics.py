"""
Evaluation metrics for image captioning: BLEU, METEOR, CIDEr
"""

import numpy as np
from collections import defaultdict
import math


def compute_bleu(references, hypothesis, n=1):
    """
    Compute BLEU-n score.
    
    Args:
        references: list of reference sentences (strings)
        hypothesis: generated sentence (string)
        n: n-gram size (1 for BLEU-1, 4 for BLEU-4)
    
    Returns:
        BLEU-n score (0-1)
    """
    # Tokenize
    hyp_tokens = hypothesis.lower().split()
    ref_tokens_list = [ref.lower().split() for ref in references]
    
    # Count n-grams
    def get_ngrams(tokens, n):
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    hyp_ngrams = get_ngrams(hyp_tokens, n)
    
    # Count matches
    max_counts = defaultdict(int)
    for ref_tokens in ref_tokens_list:
        ref_ngrams = get_ngrams(ref_tokens, n)
        for ngram in hyp_ngrams:
            max_counts[ngram] = max(max_counts[ngram], ref_ngrams[ngram])
    
    clipped_counts = {
        ngram: min(count, max_counts[ngram])
        for ngram, count in hyp_ngrams.items()
    }
    
    numerator = sum(clipped_counts.values())
    denominator = max(1, len(hyp_tokens) - n + 1)
    
    if denominator == 0:
        return 0.0
    
    precision = numerator / denominator
    
    # Brevity penalty
    hyp_len = len(hyp_tokens)
    ref_len = min([len(ref) for ref in ref_tokens_list], key=lambda x: abs(x - hyp_len))
    
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / max(1, hyp_len))
    
    return bp * precision


def compute_bleu_corpus(references_list, hypotheses_list, n=1):
    """
    Compute corpus-level BLEU-n score.
    
    Args:
        references_list: list of lists of reference sentences
        hypotheses_list: list of generated sentences
        n: n-gram size
    
    Returns:
        Average BLEU-n score
    """
    scores = []
    for refs, hyp in zip(references_list, hypotheses_list):
        if isinstance(refs, str):
            refs = [refs]
        score = compute_bleu(refs, hyp, n)
        scores.append(score)
    
    return np.mean(scores)


def compute_meteor_simple(reference, hypothesis):
    """
    Simplified METEOR score (unigram-based).
    Full METEOR requires WordNet, this is a simplified version.
    
    Args:
        reference: reference sentence (string)
        hypothesis: generated sentence (string)
    
    Returns:
        Simplified METEOR score (0-1)
    """
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())
    
    if len(hyp_tokens) == 0:
        return 0.0
    
    matches = len(ref_tokens & hyp_tokens)
    
    precision = matches / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    recall = matches / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f_mean = (precision * recall) / (precision + recall)
    
    return f_mean


def compute_meteor_corpus(references_list, hypotheses_list):
    """
    Compute corpus-level simplified METEOR score.
    
    Args:
        references_list: list of reference sentences (or lists of references)
        hypotheses_list: list of generated sentences
    
    Returns:
        Average METEOR score
    """
    scores = []
    for refs, hyp in zip(references_list, hypotheses_list):
        if isinstance(refs, list):
            # Take first reference if multiple
            ref = refs[0] if refs else ""
        else:
            ref = refs
        score = compute_meteor_simple(ref, hyp)
        scores.append(score)
    
    return np.mean(scores)


def compute_cider_simple(references_list, hypotheses_list):
    """
    Simplified CIDEr score.
    Full CIDEr requires TF-IDF computation, this is a simplified version.
    
    Args:
        references_list: list of lists of reference sentences
        hypotheses_list: list of generated sentences
    
    Returns:
        Simplified CIDEr score
    """
    def get_ngrams_all(tokens, max_n=4):
        """Get all n-grams up to max_n"""
        all_ngrams = defaultdict(int)
        for n in range(1, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                all_ngrams[ngram] += 1
        return all_ngrams
    
    scores = []
    
    for refs, hyp in zip(references_list, hypotheses_list):
        if isinstance(refs, str):
            refs = [refs]
        
        hyp_tokens = hyp.lower().split()
        hyp_ngrams = get_ngrams_all(hyp_tokens)
        
        ref_scores = []
        for ref in refs:
            ref_tokens = ref.lower().split()
            ref_ngrams = get_ngrams_all(ref_tokens)
            
            # Compute cosine similarity
            common_ngrams = set(hyp_ngrams.keys()) & set(ref_ngrams.keys())
            
            if len(common_ngrams) == 0:
                ref_scores.append(0.0)
                continue
            
            numerator = sum(hyp_ngrams[ng] * ref_ngrams[ng] for ng in common_ngrams)
            
            hyp_norm = math.sqrt(sum(v**2 for v in hyp_ngrams.values()))
            ref_norm = math.sqrt(sum(v**2 for v in ref_ngrams.values()))
            
            if hyp_norm == 0 or ref_norm == 0:
                ref_scores.append(0.0)
            else:
                ref_scores.append(numerator / (hyp_norm * ref_norm))
        
        scores.append(np.mean(ref_scores) if ref_scores else 0.0)
    
    return np.mean(scores) * 10.0  # Scale to match typical CIDEr range


def evaluate_captions(references_list, hypotheses_list):
    """
    Compute all metrics: BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, CIDEr.
    
    Args:
        references_list: list of reference sentences (or lists of references)
        hypotheses_list: list of generated sentences
    
    Returns:
        Dictionary with all metrics
    """
    # Ensure references are lists
    refs_normalized = []
    for refs in references_list:
        if isinstance(refs, str):
            refs_normalized.append([refs])
        else:
            refs_normalized.append(refs)
    
    metrics = {
        'BLEU-1': compute_bleu_corpus(refs_normalized, hypotheses_list, n=1),
        'BLEU-2': compute_bleu_corpus(refs_normalized, hypotheses_list, n=2),
        'BLEU-3': compute_bleu_corpus(refs_normalized, hypotheses_list, n=3),
        'BLEU-4': compute_bleu_corpus(refs_normalized, hypotheses_list, n=4),
        'METEOR': compute_meteor_corpus(refs_normalized, hypotheses_list),
        'CIDEr': compute_cider_simple(refs_normalized, hypotheses_list),
    }
    
    return metrics


if __name__ == "__main__":
    # Test
    references = [
        ["a dog is running in the grass", "a brown dog running"],
        ["a cat sitting on a table"],
    ]
    hypotheses = [
        "a dog running in grass",
        "a cat on table",
    ]
    
    metrics = evaluate_captions(references, hypotheses)
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
