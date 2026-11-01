"""
Bayesian Knowledge Tracing (BKT) Engine

Implements the standard Corbett & Anderson (1995) 2-state Hidden Markov Model
for tracking skill mastery over sequential learning opportunities.

Parameters per skill:
  - P(L0): Initial mastery probability P(L_1 = 1)
  - P(T):  Transition probability P(L_{t+1} = 1 | L_t = 0)
  - P(G):  Guess probability P(obs_t = 1 | L_t = 0)
  - P(S):  Slip probability P(obs_t = 0 | L_t = 1)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


class BKTModel:
    def __init__(
        self,
        p_l0: float = 0.2,
        p_t: float = 0.15,
        p_g: float = 0.25,
        p_s: float = 0.10,
    ):
        """
        Initialize BKT model with initial or fitted parameters.
        Bounds:
            p_l0 in [0.01, 0.99]
            p_t  in [0.01, 0.50]  (learning transition)
            p_g  in [0.05, 0.40]  (guessing constraint to prevent degeneracies)
            p_s  in [0.01, 0.30]  (slip constraint to prevent degeneracies)
        """
        self.p_l0 = float(np.clip(p_l0, 0.01, 0.99))
        self.p_t = float(np.clip(p_t, 0.001, 0.50))
        self.p_g = float(np.clip(p_g, 0.01, 0.45))
        self.p_s = float(np.clip(p_s, 0.01, 0.35))

    def predict_correct(self, p_l: float) -> float:
        """
        P(C_t = 1) = P(L_t) * (1 - P(S)) + (1 - P(L_t)) * P(G)
        """
        return p_l * (1.0 - self.p_s) + (1.0 - p_l) * self.p_g

    def update_mastery(self, p_l: float, correct: bool) -> float:
        """
        Forward Bayesian update after observing correctness at step t,
        followed by learning transition to step t+1.
        """
        obs = 1 if correct else 0
        if obs == 1:
            # P(L_t | obs=1) = (P(L_t) * (1 - P(S))) / P(obs=1)
            num = p_l * (1.0 - self.p_s)
            denom = num + (1.0 - p_l) * self.p_g
        else:
            # P(L_t | obs=0) = (P(L_t) * P(S)) / P(obs=0)
            num = p_l * self.p_s
            denom = num + (1.0 - p_l) * (1.0 - self.p_g)

        p_posterior = num / (denom + 1e-12)
        p_posterior = float(np.clip(p_posterior, 0.0, 1.0))

        # Transition to next step: P(L_{t+1}) = P(L_t | obs) + (1 - P(L_t | obs)) * P(T)
        p_next = p_posterior + (1.0 - p_posterior) * self.p_t
        return float(np.clip(p_next, 0.0, 1.0))

    def forward_sequence(self, sequence: List[int]) -> Tuple[List[float], List[float]]:
        """
        Given a sequence of binary responses [obs_1, obs_2, ..., obs_T],
        returns:
            predictions: P(C_t = 1) before observing obs_t
            masteries: P(L_t) after updating with obs_t
        """
        predictions = []
        masteries = []
        current_l = self.p_l0

        for obs in sequence:
            p_c = self.predict_correct(current_l)
            predictions.append(p_c)
            current_l = self.update_mastery(current_l, bool(obs))
            masteries.append(current_l)

        return predictions, masteries

    def fit_em(
        self,
        sequences: List[List[int]],
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Fit parameters [p_l0, p_t, p_g, p_s] across a collection of student sequences
        using Baum-Welch (Expectation-Maximization for 2-state HMM).
        """
        # Filter empty sequences
        seqs = [s for s in sequences if len(s) > 0]
        if not seqs:
            return self.get_params()

        p_l0 = self.p_l0
        p_t = self.p_t
        p_g = self.p_g
        p_s = self.p_s

        prev_ll = -np.inf

        for iteration in range(max_iter):
            # Transition matrix A:
            # A[0,0] = 1 - p_t, A[0,1] = p_t
            # A[1,0] = 0.0    , A[1,1] = 1.0 (absorbing or standard BKT no unlearning)
            A = np.array([[1.0 - p_t, p_t], [0.0, 1.0]])

            # Emission matrix B: B[state, obs] where obs in {0, 1}
            # state 0: unmastered, state 1: mastered
            # B[0, 1] = p_g, B[0, 0] = 1 - p_g
            # B[1, 1] = 1 - p_s, B[1, 0] = p_s
            B = np.array([[1.0 - p_g, p_g], [p_s, 1.0 - p_s]])

            pi = np.array([1.0 - p_l0, p_l0])

            sum_gamma_0 = 0.0
            sum_xi_01 = 0.0
            sum_xi_00 = 0.0
            sum_gamma_0_obs1 = 0.0
            sum_gamma_0_all = 0.0
            sum_gamma_1_obs0 = 0.0
            sum_gamma_1_all = 0.0

            total_ll = 0.0

            for seq in seqs:
                T = len(seq)
                # Forward pass: alpha[t, state]
                alpha = np.zeros((T, 2))
                scale = np.zeros(T)

                alpha[0] = pi * B[:, seq[0]]
                scale[0] = np.sum(alpha[0]) + 1e-12
                alpha[0] /= scale[0]

                for t in range(1, T):
                    alpha[t] = np.dot(alpha[t - 1], A) * B[:, seq[t]]
                    scale[t] = np.sum(alpha[t]) + 1e-12
                    alpha[t] /= scale[t]

                total_ll += np.sum(np.log(scale))

                # Backward pass: beta[t, state]
                beta = np.zeros((T, 2))
                beta[T - 1] = 1.0

                for t in range(T - 2, -1, -1):
                    beta[t] = np.dot(A, (B[:, seq[t + 1]] * beta[t + 1]))
                    beta[t] /= scale[t + 1] + 1e-12

                # Gamma: posterior marginal state probability
                gamma = alpha * beta
                gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-12

                # Accumulate for L0
                sum_gamma_0 += gamma[0, 1]

                # Accumulate for emissions
                for t in range(T):
                    obs = seq[t]
                    sum_gamma_0_all += gamma[t, 0]
                    sum_gamma_1_all += gamma[t, 1]
                    if obs == 1:
                        sum_gamma_0_obs1 += gamma[t, 0]
                    else:
                        sum_gamma_1_obs0 += gamma[t, 1]

                # Accumulate for transition (Xi)
                for t in range(T - 1):
                    denom = (
                        np.sum(
                            alpha[t, :, None] * A * B[:, seq[t + 1]] * beta[t + 1, :]
                        )
                        + 1e-12
                    )
                    xi = (
                        alpha[t, :, None] * A * B[:, seq[t + 1]] * beta[t + 1, :]
                    ) / denom
                    sum_xi_01 += xi[0, 1]
                    sum_xi_00 += xi[0, 0]

            # M-step update with bounded constraints to prevent degeneracy
            n_seqs = len(seqs)
            p_l0 = np.clip(sum_gamma_0 / n_seqs, 0.05, 0.85)

            if (sum_xi_00 + sum_xi_01) > 1e-9:
                p_t = np.clip(sum_xi_01 / (sum_xi_00 + sum_xi_01), 0.01, 0.40)

            if sum_gamma_0_all > 1e-9:
                p_g = np.clip(sum_gamma_0_obs1 / sum_gamma_0_all, 0.05, 0.40)

            if sum_gamma_1_all > 1e-9:
                p_s = np.clip(sum_gamma_1_obs0 / sum_gamma_1_all, 0.01, 0.30)

            if abs(total_ll - prev_ll) < tol:
                break
            prev_ll = total_ll

        self.p_l0 = float(p_l0)
        self.p_t = float(p_t)
        self.p_g = float(p_g)
        self.p_s = float(p_s)
        return self.get_params()

    def get_params(self) -> Dict[str, float]:
        return {
            "p_l0": round(self.p_l0, 4),
            "p_t": round(self.p_t, 4),
            "p_g": round(self.p_g, 4),
            "p_s": round(self.p_s, 4),
        }

    def evaluate_metrics(self, sequences: List[List[int]]) -> Dict[str, float]:
        """
        Evaluate predictive accuracy against ground-truth binary responses.
        Calculates AUC-ROC, Log-Loss, Brier Score, and Calibration Error.
        """
        all_true = []
        all_pred = []

        for seq in sequences:
            if not seq:
                continue
            preds, _ = self.forward_sequence(seq)
            all_true.extend(seq)
            all_pred.extend(preds)

        y_true = np.array(all_true)
        y_pred = np.array(all_pred)

        if len(np.unique(y_true)) < 2:
            return {"error": "Need both correct and incorrect observations for AUC"}

        auc = float(roc_auc_score(y_true, y_pred))
        brier = float(brier_score_loss(y_true, y_pred))
        ll = float(log_loss(y_true, y_pred, eps=1e-7))

        # Expected Calibration Error (ECE) with 10 bins
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for i in range(10):
            mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
            if np.any(mask):
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_pred[mask])
                ece += np.sum(mask) / len(y_pred) * np.abs(bin_acc - bin_conf)

        return {
            "auc_roc": round(auc, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(ll, 4),
            "ece": round(float(ece), 4),
            "n_observations": len(y_true),
        }
