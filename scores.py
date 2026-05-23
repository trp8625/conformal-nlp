"""
scores.py — Nonconformity score functions for conformal-nlp.

Each score function takes:
    probs : np.ndarray of shape (n_examples, n_classes)
        Softmax probabilities from a classifier.
    labels : np.ndarray of shape (n_examples,)
        Integer true class indices.

And returns:
    scores : np.ndarray of shape (n_examples,)
        A nonconformity score per example. Higher = more nonconforming
        (i.e. the model was more "surprised" by the true label).

At prediction time, each function also has a companion `_threshold_mask`
that, given probs for a single example and a threshold q_hat, returns a
boolean mask over classes indicating which are included in the prediction set.
"""

import numpy as np



# LAC — Least Ambiguous Classifier (simplest, softmax-based)

def lac_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    LAC nonconformity score: 1 - p(true label).

    Intuition: the score is low when the model is confident in the right
    answer, and high when it isn't. Simple and fast; works well for
    balanced binary/small-class problems. Can produce large sets on
    many-class imbalanced tasks because rare classes always get low
    softmax mass.

    Args:
        probs:  (n, k) softmax probabilities.
        labels: (n,)   integer true class indices in [0, k).

    Returns:
        scores: (n,) nonconformity scores in [0, 1].
    """
    n = len(labels)
    true_probs = probs[np.arange(n), labels]   # p(y_true) for each example
    return 1.0 - true_probs


def lac_predict(probs: np.ndarray, q_hat: float) -> np.ndarray:
    """
    Build prediction sets using the LAC threshold.

    Include class c if: 1 - p(c) <= q_hat
    Equivalently:        p(c) >= 1 - q_hat

    Args:
        probs: (n, k) softmax probabilities for n test examples.
        q_hat: scalar threshold computed during calibration.

    Returns:
        sets: (n, k) boolean array. sets[i, c] = True means class c
              is included in the prediction set for example i.
    """
    return probs >= (1.0 - q_hat)



# APS — Adaptive Prediction Sets

def aps_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    APS nonconformity score.

    Sort classes by descending softmax probability. Accumulate probability
    mass until you reach the true label. The score is the cumulative sum
    up to and including the true label's position.

    Formally:
        s_APS(x, y) = sum_{j: rank(j) <= rank(y)} p(class_j)

    where rank is determined by descending softmax order.

    Intuition: if the true label is always ranked first (model is sure),
    s is just p(top class) ≈ high, but the threshold will also be high —
    the important thing is relative ordering. When the true label is buried
    deep in the ranking, s is close to 1.

    APS handles class imbalance better than LAC because it accounts for
    how many classes the model has to "pass through" before reaching the
    true label.

    Args:
        probs:  (n, k) softmax probabilities.
        labels: (n,)   integer true class indices.

    Returns:
        scores: (n,) nonconformity scores in (0, 1].
    """
    n, k = probs.shape
    scores = np.zeros(n)

    for i in range(n):
        # Descending sort of class indices by probability
        sorted_indices = np.argsort(probs[i])[::-1]   # shape (k,)
        sorted_probs   = probs[i][sorted_indices]       # shape (k,)
        cumsum         = np.cumsum(sorted_probs)        # shape (k,)

        # Find where the true label lands in the sorted order
        true_rank = np.where(sorted_indices == labels[i])[0][0]  # scalar
        scores[i] = cumsum[true_rank]

    return scores


def aps_predict(probs: np.ndarray, q_hat: float) -> np.ndarray:
    """
    Build prediction sets using APS threshold.

    Include classes greedily in descending probability order until the
    cumulative sum exceeds q_hat. Always include at least the top class.

    Args:
        probs: (n, k) softmax probabilities.
        q_hat: scalar calibration threshold.

    Returns:
        sets: (n, k) boolean array.
    """
    n, k = probs.shape
    sets = np.zeros((n, k), dtype=bool)

    for i in range(n):
        sorted_indices = np.argsort(probs[i])[::-1]
        sorted_probs   = probs[i][sorted_indices]
        cumsum         = np.cumsum(sorted_probs)

        # Include all classes up to (and including) the one that pushes
        # cumsum over q_hat; always include at least the top-1 class.
        cutoff = np.searchsorted(cumsum, q_hat, side='left')
        cutoff = min(cutoff + 1, k)   # +1 to include the crossing class

        included = sorted_indices[:cutoff]
        sets[i, included] = True

    return sets


# RAPS — Regularized Adaptive Prediction Sets
def raps_scores(
    probs: np.ndarray,
    labels: np.ndarray,
    lam: float = 0.1,
    k_reg: int = 1,
) -> np.ndarray:
    """
    RAPS nonconformity score.

    Adds a regularization penalty to APS to discourage large prediction sets.

    Formally:
        s_RAPS(x, y) = s_APS(x, y) + lambda * max(rank(y) - k_reg, 0)

    The penalty term kicks in when the true label's rank exceeds k_reg.
    This pressures the calibrated threshold to be tighter, shrinking sets
    on easy examples. Most useful on many-class imbalanced problems like
    GoEmotions where APS alone still produces large sets.

    Args:
        probs:  (n, k) softmax probabilities.
        labels: (n,)   integer true class indices.
        lam:    regularization strength (lambda). Larger = more aggressive
                set size reduction. You may want to verify a good default
                for your task; 0.1 is a reasonable starting point but is
                not universally optimal.
        k_reg:  rank threshold below which no penalty applies. A value of
                1 means the top-1 class is never penalized.

    Returns:
        scores: (n,) nonconformity scores.
    """
    n, k = probs.shape
    scores = np.zeros(n)

    for i in range(n):
        sorted_indices = np.argsort(probs[i])[::-1]
        sorted_probs   = probs[i][sorted_indices]
        cumsum         = np.cumsum(sorted_probs)

        true_rank = np.where(sorted_indices == labels[i])[0][0]

        aps_component  = cumsum[true_rank]
        raps_penalty   = lam * max(true_rank - k_reg + 1, 0)
        scores[i]      = aps_component + raps_penalty

    return scores


def raps_predict(
    probs: np.ndarray,
    q_hat: float,
    lam: float = 0.1,
    k_reg: int = 1,
) -> np.ndarray:
    """
    Build prediction sets using RAPS threshold.

    Include classes greedily (descending prob order), accumulating the
    RAPS score until it exceeds q_hat.

    Args:
        probs: (n, k) softmax probabilities.
        q_hat: scalar calibration threshold.
        lam:   must match the value used during calibration.
        k_reg: must match the value used during calibration.

    Returns:
        sets: (n, k) boolean array.
    """
    n, k = probs.shape
    sets = np.zeros((n, k), dtype=bool)

    for i in range(n):
        sorted_indices = np.argsort(probs[i])[::-1]
        sorted_probs   = probs[i][sorted_indices]
        cumsum         = np.cumsum(sorted_probs)

        for rank in range(k):
            penalty      = lam * max(rank - k_reg + 1, 0)
            raps_val     = cumsum[rank] + penalty
            sets[i, sorted_indices[rank]] = True
            if raps_val >= q_hat:
                break

    return sets



# Registry — lets calibration.py look up scores by name

SCORE_FN = {
    "LAC":  lac_scores,
    "APS":  aps_scores,
    "RAPS": raps_scores,
}

PREDICT_FN = {
    "LAC":  lac_predict,
    "APS":  aps_predict,
    "RAPS": raps_predict,
}
