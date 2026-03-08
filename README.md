# Sleep Breathing Irregularity Detection
### SRIP 2026 — AI for Health & Wellbeing

> **Completed with AI assistance.**
> Built with **Claude Sonnet 4.6** (Anthropic, 2025) for pipeline design, code generation, and report drafting. All training and results were executed locally by the student.
>
> **Stack:** Python 3.13 · PyTorch · Scikit-learn · Pandas · NumPy · Matplotlib

---

Automated detection of breathing irregularities (Hypopnea, Obstructive Apnea) from 8-hour overnight PSG recordings using a 1D CNN trained directly on raw physiological signals.

---

## File Structure

```
SRIP_Health/
├── Data/
│   ├── AP01/                              # Raw .txt files for participant 01
│   ├── AP02/
│   ├── AP03/
│   ├── AP04/
│   └── AP05/
│
├── Dataset/
│   ├── combined_dataset.csv               # Main dataset — all participants merged
│   └── lopo_results_cnn.csv               # Per-fold metrics table
│
├── Results/
│   ├── confusion_matrix_cnn.png           # Overall confusion matrix
│   └── per_fold_accuracy_cnn.png          # Per-fold accuracy bar chart
│
├── Visualizations/
│   └── <PID>_visualization.pdf            # Annotated signal PDFs
│
├── models/
│   └── cnn_model.py                       # BreathingCNN class
│
├── scripts/
│   ├── 01_build_combined_dataset.ipynb    # Parse + combine all signals
│   ├── 02_visualize_combined.ipynb        # Annotated PDF visualisation
│   └── 03_train_cnn_kaggle.ipynb          # Train and evaluate CNN (Kaggle GPU)
│
├── requirements.txt
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Participants | 5 (AP01–AP05) |
| Recording | ~8 hours overnight |
| Signals | Nasal Airflow (32 Hz), Thoracic Movement (32 Hz), SpO2 (4 Hz) |
| Event labels | Normal, Hypopnea, Obstructive Apnea |
| Sleep stages | Wake, N1, N2, N3, REM, Movement |

### `combined_dataset.csv` columns

| Column | Type | Description |
|---|---|---|
| `datetime` | datetime | Sample timestamp at 32 Hz |
| `participant` | string | AP01–AP05 |
| `sleep_stage` | string | Wake / N1 / N2 / N3 / REM / Movement |
| `event_label` | string | Normal / Hypopnea / Obstructive Apnea |
| `nasal` | float | Raw nasal airflow amplitude |
| `spo2` | float | SpO2 percentage |
| `thoracic` | float | Raw thoracic movement amplitude |

**Upsampling:** SpO2 (4 Hz) is upsampled to 32 Hz by repeating each value 8×. Sleep profile (one epoch/30 sec) is upsampled via `np.searchsorted` — each sample inherits the stage of the last epoch started before it. Body event and Mixed Apnea are merged into Obstructive Apnea (physiologically similar, too few samples to learn separately).

---

## Installation

```bash
pip install -r requirements.txt
```

```
# requirements.txt
numpy
pandas
torch
scikit-learn
matplotlib
```

> Raw data files are not included due to size. Place participant folders `AP01`–`AP05` under `Data/` before running.

---

## Usage — run notebooks in order

### 1. Build combined dataset
```
scripts/01_build_combined_dataset.ipynb
```
Parses all raw `.txt` files, upsamples all signals to 32 Hz, assigns sleep stage and event label to every sample, saves `Dataset/combined_dataset.csv`.

### 2. Visualise
```
scripts/02_visualize_combined.ipynb
```
Generates one PDF per participant (~96 pages, 5 min/page). Each page shows a sleep stage colour strip, 3 signal panels with coloured event shading, and a bottom event timeline. X-axis ticks every 5 seconds.

### 3. Train CNN (Kaggle GPU)
```
scripts/03_train_cnn_kaggle.ipynb
```
Upload `combined_dataset.csv` to a Kaggle dataset. Enable GPU T4. The notebook auto-detects the dataset path, writes the model inline, cuts 15-second windows, and runs LOPO-CV. Download output files to `Results/` and `Dataset/`.

---

## Model — BreathingCNN

3-block 1D CNN in `models/cnn_model.py`. Operates directly on raw signal windows — no hand-crafted features.

```
Input: (batch, 3, 480)
  3 channels = nasal / spo2 / thoracic
  480 samples = 15 seconds at 32 Hz

Block A: Conv1d(3→32,  k=7) + BN + ReLU + MaxPool(2)      — local patterns (~220ms)
Block B: Conv1d(32→64, k=5) + BN + ReLU + MaxPool(2)      — pattern combinations
Block C: Conv1d(64→128,k=3) + BN + ReLU + AdaptiveAvgPool  — global window summary

Classifier: Flatten → Linear(128→64) → ReLU → Dropout → Linear(64→n_classes)
Output: (batch, n_classes)   ~44K parameters
```

**Training details:**
- Loss: Focal Loss (γ=2) — down-weights easy Normal windows, focuses on hard event windows
- Optimiser: AdamW (lr=5×10⁻⁴, weight decay=1×10⁻³)
- Scheduler: CosineAnnealingLR (50 epochs, η_min = lr/20)
- Normalisation: per-channel z-score fitted on training fold only (no data leakage)
- Window: 15s, step 7s (~53% overlap), event label if ≥40% of window is that event

---

## Results — LOPO-CV (1D CNN, Focal Loss)

### Per-fold accuracy

| Fold | Test Participant | Windows | Accuracy |
|---|---|---|---|
| 1 | AP01 | 3,905 | 0.8251 |
| 2 | AP02 | 3,792 | 0.8584 |
| 3 | AP03 | 3,635 | 0.9164 |
| 4 | AP04 | 4,142 | 0.8452 |
| 5 | AP05 | 3,390 | 0.7150 |
| **Mean** | — | **18,864** | **0.8320 ± 0.072** |

### Per-class performance (overall)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Hypopnea | 0.16–0.75 | 0.01–0.13 | 0.02–0.20 | 2,141 |
| Normal | 0.73–0.98 | 0.90–1.00 | 0.83–0.96 | 16,128 |
| Obstructive Apnea | 0.00–0.60 | 0.00–0.40 | 0.00–0.29 | 595 |

### Key observations

- **Normal** is classified reliably across all folds (F1 0.83–0.96), reflecting the class dominance (~85% of windows).
- **Hypopnea recall is low** (1–13%) — the model detects some Hypopnea in AP01 but misses most in other folds. The main bottleneck is the large Normal:Hypopnea imbalance (~8:1 in training windows).
- **AP03** reaches the highest accuracy (91.6%) but AP03 has the fewest events (28 total), so most test windows are Normal — the model scores well by predicting Normal.
- **AP05** is the hardest fold (71.5%) because AP05 has the highest event rate (~25%) including heavy Obstructive Apnea (459 test windows), which the model sees very little of in training from other participants.
- Focal Loss (vs standard CrossEntropy) stabilised training and eliminated the catastrophic collapse seen in earlier runs where the model predicted a single class for entire folds.

---

## Configuration

| Parameter | Value | Description |
|---|---|---|
| `WIN_SEC` | 15 | Window size in seconds |
| `STEP_SEC` | 7 | Step size (~53% overlap) |
| `OV_THR` | 0.4 | Min event overlap fraction to label a window |
| `EPOCHS` | 50 | Training epochs per fold |
| `BATCH_SIZE` | 128 | Mini-batch size |
| `LR` | 5×10⁻⁴ | AdamW learning rate |
| `DROPOUT` | 0.4 | Dropout probability in classifier |

---

## Author

**Syed Mohd Tashif**
B.Tech Artificial Intelligence, ZHCET AMU Aligarh
[syedtashif239@gmail.com](mailto:syedtashif239@gmail.com) · [GitHub](https://github.com/syed-tashif) · [LinkedIn](https://linkedin.com/in/syed-tashif)
