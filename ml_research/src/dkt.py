"""
Deep Knowledge Tracing (DKT) Neural Baseline

PyTorch implementation of Deep Knowledge Tracing (Piech et al., 2015).
Uses an LSTM/GRU recurrent network to predict student performance on future questions
given the history of past skill interactions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict


class DKTNet(nn.Module):
    def __init__(
        self,
        num_skills: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super(DKTNet, self).__init__()
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim
        # Input dimension: 2 * num_skills (skill_id + binary correctness encoding)
        self.input_dim = 2 * num_skills

        self.rnn = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_skills)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, 2 * num_skills]
        output shape: [batch_size, seq_len, num_skills]
        """
        out, _ = self.rnn(x)
        logits = self.fc(out)
        preds = self.sigmoid(logits)
        return preds


def encode_interactions(
    interactions: List[Tuple[int, int]],
    num_skills: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode sequence of (skill_id, correctness) tuples into one-hot tensors.
    interactions: list of (skill_id, is_correct) where skill_id in [0, num_skills-1]
    returns: (x_encoded, target_skills, target_correct)
    """
    seq_len = len(interactions)
    x = np.zeros((seq_len, 2 * num_skills), dtype=np.float32)
    target_skills = np.zeros(seq_len, dtype=np.int64)
    target_correct = np.zeros(seq_len, dtype=np.float32)

    for t, (skill_id, correct) in enumerate(interactions):
        # Input encoding: index = skill_id + correct * num_skills
        offset = num_skills if correct == 1 else 0
        x[t, skill_id + offset] = 1.0
        target_skills[t] = skill_id
        target_correct[t] = float(correct)

    return x, target_skills, target_correct
