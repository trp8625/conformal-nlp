"""
calibration.py — Split conformal calibration for conformal-nlp.

The single public function `calibrate` takes nonconformity scores from a
held-out calibration set and returns q_hat: the threshold you'll use at
prediction time.

Math recap:
    Given n calibration scores s_1, ..., s_n and a target miscoverage
    rate alpha (e.g. 0.1 for 90% coverage):

        q_hat = the ceil((n+1)(1-alpha) / n)-th quantile of the scores

    The (n+1) correction — rather than just n — is what makes coverage
    exact in finite samples rather than approximate. For large n it barely
    changes the result, but it's the mathematically correct thing to do.

    Guarantee: P(y_true in prediction_set) >= 1 - alpha,
    under the assumption that calibration and test examples are
    exchangeable (i.e. drawn i.i.d. from the same distribution).
"""

import numpy as np


def calibrate(
    scores: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Compute the conformal calibration threshold q_hat.

    Args:
        scores: (n,) nonconformity scores from the calibration set.
                Produced by one of the score functions in scores.py.
        alpha:  target miscoverage rate in (0, 1). E.g. 0.1 gives a
                90% coverage guarantee.

    Returns:
        q_hat:  scalar float. The threshold to use at prediction time.
                A test example's score <= q_hat means that label is
                included in the prediction set.

    Raises:
        ValueError: if alpha is not in (0, 1), or scores is empty.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(scores)
    if n == 0:
        raise ValueError("scores array is empty — pass a non-empty calibration set.")

    # Finite-sample corrected quantile level
    # ceil ensures we round up, which keeps coverage >= 1-alpha
    quantile_level = np.ceil((n + 1) * (1 - alpha)) / n

    # Clip to [0, 1] — if alpha is very small and n is tiny, the level
    # can exceed 1.0, in which case we return the max score (most conservative).
    quantile_level = min(quantile_level, 1.0)

    q_hat = float(np.quantile(scores, quantile_level))
    return q_hat


def coverage(
    prediction_sets: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute empirical marginal coverage.

    The fraction of test examples where the true label is in the
    prediction set. Should be >= 1-alpha when calibration worked correctly.

    Args:
        prediction_sets: (n, k) boolean array from a predict function.
        labels:          (n,) integer true class indices.

    Returns:
        Scalar float in [0, 1].
    """
    n = len(labels)
    covered = sum(
        prediction_sets[i, labels[i]]
        for i in range(n)
    )
    return covered / n


def average_set_size(prediction_sets: np.ndarray) -> float:
    """
    Compute mean prediction set size (efficiency metric).

    Smaller is better, as long as coverage is maintained. A set size of 1
    on every example means the model is perfectly confident and correct.
    A set size equal to the number of classes means the model learned nothing.

    Args:
        prediction_sets: (n, k) boolean array.

    Returns:
        Scalar float.
    """
    return float(prediction_sets.sum(axis=1).mean())
