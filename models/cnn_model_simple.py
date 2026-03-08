"""
models/cnn_model.py  —  BreathingCNN (Simple)

ARCHITECTURE — one clean forward pass, nothing fancy
──────────────────────────────────────────────────────

INPUT
  Shape : (batch, 3, 480)
  3 channels = nasal airflow / SpO2 / thoracic movement
  480 samples = 15 seconds × 32 Hz

STEP 1 — Block A  "What is happening locally?"
  Conv1d(3 → 32, kernel=7)
  kernel=7 covers ~220ms of signal at 32 Hz.
  Think of it as a small sliding template — the kernel fires
  when it finds a matching shape in the signal.
  32 different templates are learned simultaneously.
  → BatchNorm + ReLU + MaxPool(2)  [time dimension halved]

STEP 2 — Block B  "Do local patterns combine into something?"
  Conv1d(32 → 64, kernel=5)
  Each new filter sees all 32 outputs from Block A.
  It learns combinations: e.g.
    "flat nasal (no airflow) AND active thoracic (chest still trying)"
    → likely Obstructive Apnea
  → BatchNorm + ReLU + MaxPool(2)  [time halved again]

STEP 3 — Block C  "What is the overall picture in this window?"
  Conv1d(64 → 128, kernel=3)
  AdaptiveAvgPool(1) collapses ALL remaining time steps into
  a single value per filter → one 128-dim vector per window.
  This is the model's fixed-size summary of the 15-second clip.

STEP 4 — Classifier  "Which class?"
  Linear(128 → 64) → ReLU → Dropout → Linear(64 → n_classes)
  Two FC layers map the 128-dim summary to n_classes scores.

OUTPUT
  Shape : (batch, n_classes)  — raw logits
  Pass through softmax to get probabilities.

PARAMETER COUNT  ~98K
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch,
                      kernel_size=kernel,
                      padding=kernel // 2),  # padding keeps length unchanged
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(pool),              # halve the time dimension
        )

    def forward(self, x):
        return self.block(x)


class BreathingCNN(nn.Module):

    def __init__(self, n_classes: int, window_samples: int, dropout: float = 0.4):
        super().__init__()

        self.block_a = ConvBlock(  3,  32, kernel=7)   # local patterns
        self.block_b = ConvBlock( 32,  64, kernel=5)   # pattern combinations

        self.block_c = nn.Sequential(                  # global summary
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                   # (batch, 128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                              # (batch, 128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.block_a(x)         # (batch,  32, T/2)
        x = self.block_b(x)         # (batch,  64, T/4)
        x = self.block_c(x)         # (batch, 128,   1)
        return self.classifier(x)   # (batch, n_classes)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
