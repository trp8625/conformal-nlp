"""
wrapper.py — ConformalClassifier: the main user-facing class for conformal-nlp.

Wraps any HuggingFace AutoModelForSequenceClassification model and adds
split conformal calibration, producing prediction sets with coverage guarantees
instead of point estimates.

Typical usage:
    from conformal_nlp.wrapper import ConformalClassifier

    cc = ConformalClassifier(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        score="LAC",
        alpha=0.1,
    )
    cc.calibrate(cal_texts, cal_labels)
    prediction_sets = cc.predict(test_texts)
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional

from scores import SCORE_FN, PREDICT_FN, raps_scores, raps_predict
from calibration import calibrate, coverage, average_set_size


class ConformalClassifier:
    """
    A conformal wrapper around any HuggingFace sequence classification model.

    After calling `.calibrate()` on a held-out set, `.predict()` returns
    prediction sets — subsets of class labels — that contain the true label
    with probability >= 1 - alpha.

    Args:
        model_name: HuggingFace model hub name or local path.
                    Must be a sequence classification model
                    (AutoModelForSequenceClassification compatible).
        score:      Nonconformity score to use. One of "LAC", "APS", "RAPS".
                    RAPS is the most rigorous; LAC is fastest to reason about.
        alpha:      Target miscoverage rate. 0.1 = 90% coverage guarantee.
        device:     "cuda", "cpu", or None (auto-detects).
        batch_size: How many texts to encode at once. Tune down if you hit
                    OOM errors on GPU.
        raps_lam:   Lambda for RAPS regularization. Only used if score="RAPS".
        raps_k_reg: k_reg for RAPS. Only used if score="RAPS".
    """

    def __init__(
        self,
        model_name: str,
        score: str = "RAPS",
        alpha: float = 0.1,
        device: Optional[str] = None,
        batch_size: int = 32,
        raps_lam: float = 0.1,
        raps_k_reg: int = 1,
    ):
        if score not in SCORE_FN:
            raise ValueError(
                f"score must be one of {list(SCORE_FN.keys())}, got '{score}'"
            )
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.model_name  = model_name
        self.score       = score
        self.alpha       = alpha
        self.batch_size  = batch_size
        self.raps_lam    = raps_lam
        self.raps_k_reg  = raps_k_reg

        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        print(f"Loading model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Label map: integer index -> label string (e.g. {0: "NEGATIVE", 1: "POSITIVE"})
        self.id2label = self.model.config.id2label

        # Set after calibration
        self.q_hat: Optional[float] = None
        self._cal_scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal: encode texts -> softmax probabilities
    # ------------------------------------------------------------------

    def _get_probs(self, texts: List[str]) -> np.ndarray:
        """
        Run texts through the model in batches and return softmax probabilities.

        Args:
            texts: list of raw strings.

        Returns:
            probs: (n, k) numpy array of softmax probabilities.
        """
        all_probs = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = self.model(**encoded).logits   # (batch, k)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)   # (n, k)

    # ------------------------------------------------------------------
    # Public: calibrate
    # ------------------------------------------------------------------

    def calibrate(
        self,
        texts: List[str],
        labels: List[int],
    ) -> float:
        """
        Run split conformal calibration on a held-out calibration set.

        This must be called before .predict(). The calibration set should
        be separate from both training data and the test set you'll evaluate on.

        Args:
            texts:  list of raw strings (calibration examples).
            labels: list of integer class indices corresponding to each text.
                    Must match the model's label indexing (check model.config.id2label).

        Returns:
            q_hat: the computed calibration threshold (also stored as self.q_hat).
        """
        labels_arr = np.array(labels)
        probs      = self._get_probs(texts)

        # Compute nonconformity scores
        if self.score == "RAPS":
            cal_scores = raps_scores(
                probs, labels_arr,
                lam=self.raps_lam,
                k_reg=self.raps_k_reg,
            )
        else:
            cal_scores = SCORE_FN[self.score](probs, labels_arr)

        self._cal_scores = cal_scores
        self.q_hat       = calibrate(cal_scores, alpha=self.alpha)

        print(
            f"Calibrated | score={self.score} | alpha={self.alpha} "
            f"| n_cal={len(texts)} | q_hat={self.q_hat:.4f}"
        )
        return self.q_hat

    # ------------------------------------------------------------------
    # Public: predict
    # ------------------------------------------------------------------

    def predict(
        self,
        texts: List[str],
    ) -> List[set]:
        """
        Produce prediction sets for a list of texts.

        Each prediction set is a Python set of label strings (not integers),
        e.g. {"POSITIVE"} or {"POSITIVE", "NEGATIVE"}.

        A set size of 1 means the model is confident. A larger set means
        the model is uncertain and the set expands to maintain coverage.

        Args:
            texts: list of raw strings to predict on.

        Returns:
            prediction_sets: list of sets of label strings, one per input text.

        Raises:
            RuntimeError: if .calibrate() has not been called yet.
        """
        if self.q_hat is None:
            raise RuntimeError(
                "Model has not been calibrated. Call .calibrate(cal_texts, cal_labels) first."
            )

        probs = self._get_probs(texts)

        if self.score == "RAPS":
            bool_sets = raps_predict(
                probs, self.q_hat,
                lam=self.raps_lam,
                k_reg=self.raps_k_reg,
            )
        else:
            bool_sets = PREDICT_FN[self.score](probs, self.q_hat)

        # Convert boolean arrays to sets of label strings
        prediction_sets = []
        for i in range(len(texts)):
            label_set = {
                self.id2label[j]
                for j in range(bool_sets.shape[1])
                if bool_sets[i, j]
            }
            prediction_sets.append(label_set)

        return prediction_sets

    # ------------------------------------------------------------------
    # Public: evaluate (convenience method for test sets)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        texts: List[str],
        labels: List[int],
    ) -> dict:
        """
        Predict on a labeled test set and return coverage + efficiency metrics.

        Args:
            texts:  list of raw strings.
            labels: list of integer true class indices.

        Returns:
            dict with keys:
                "coverage"      : empirical marginal coverage (should be >= 1-alpha)
                "avg_set_size"  : mean prediction set size
                "q_hat"         : the threshold used
                "alpha"         : the target miscoverage rate
                "score"         : the nonconformity score used
        """
        if self.q_hat is None:
            raise RuntimeError(
                "Model has not been calibrated. Call .calibrate() first."
            )

        labels_arr = np.array(labels)
        probs      = self._get_probs(texts)

        if self.score == "RAPS":
            bool_sets = raps_predict(
                probs, self.q_hat,
                lam=self.raps_lam,
                k_reg=self.raps_k_reg,
            )
        else:
            bool_sets = PREDICT_FN[self.score](probs, self.q_hat)

        return {
            "coverage"     : coverage(bool_sets, labels_arr),
            "avg_set_size" : average_set_size(bool_sets),
            "q_hat"        : self.q_hat,
            "alpha"        : self.alpha,
            "score"        : self.score,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        calibrated = f"q_hat={self.q_hat:.4f}" if self.q_hat is not None else "not calibrated"
        return (
            f"ConformalClassifier("
            f"model='{self.model_name}', "
            f"score={self.score}, "
            f"alpha={self.alpha}, "
            f"{calibrated})"
        )
