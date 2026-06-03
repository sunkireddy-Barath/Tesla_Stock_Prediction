<div align="center">

# Tesla Stock Price Prediction using LSTM

**Deep Learning · Time-Series Forecasting · Financial AI**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.5.0-red?style=flat-square&logo=keras)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Yahoo_Finance-purple?style=flat-square)](https://finance.yahoo.com/quote/TSLA)

> Predicting Tesla (TSLA) closing prices using a Long Short-Term Memory (LSTM) neural network trained on 13 years of real historical market data — from IPO (June 2010) through December 2023.

**GitHub:** [sunkireddy-Barath](https://github.com/sunkireddy-Barath)

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Pipeline Architecture](#pipeline-architecture)
- [LSTM Model Architecture](#lstm-model-architecture)
- [LSTM Cell — Internal Mechanics](#lstm-cell--internal-mechanics)
- [Windowing Architecture](#windowing-architecture)
- [Train / Validation / Test Split](#train--validation--test-split)
- [Training Loop Architecture](#training-loop-architecture)
- [Recursive Forecasting Architecture](#recursive-forecasting-architecture)
- [Parameter Count Breakdown](#parameter-count-breakdown)
- [Stage-by-Stage Code Walkthrough](#stage-by-stage-code-walkthrough)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Results and Observations](#results-and-observations)
- [Why This Approach is Different](#why-this-approach-is-different)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Conclusion](#conclusion)

---

## Project Overview

Tesla Inc. (TSLA) is one of the most volatile and actively traded stocks on NASDAQ, with prices ranging from **$1.59 at IPO (2010)** to over **$400 at peak (2021)**. Predicting such non-linear, time-dependent price movements requires a model that can capture **long-range temporal dependencies** — something that traditional statistical models like ARIMA and basic regression fundamentally cannot do.

This project applies a **Long Short-Term Memory (LSTM)** deep learning network — a specialized form of Recurrent Neural Network (RNN) designed to learn and remember patterns across time — to forecast daily closing prices using only historical price data.

### Key Highlights

| Feature | Detail |
|---|---|
| Dataset | 3,400 trading days (June 2010 – December 2023) |
| Model Type | LSTM Neural Network (Univariate Time-Series) |
| Prediction Target | Daily Closing Price (USD) |
| Window Size | 3 prior trading days → predict next day |
| Data Split | 80% Train / 10% Validation / 10% Test (temporal) |
| Total Parameters | 20,065 (~78 KB model) |
| Forecast Types | Standard predictions + 14-step Recursive forecast |
| Framework | TensorFlow 2.17 + Keras 3.5 |

---

## Project Structure

```
Tesla_Stock_Prediction/
│
├── TSLA.csv                    Raw dataset from Yahoo Finance (3,400 rows)
├── TSLA_Stock_LSTM.ipynb       Main Jupyter Notebook — complete ML pipeline
├── TSLA_Stock_LSTM.html        Rendered HTML output for sharing/review
├── requirements.txt            Pinned library versions for reproducibility
└── README.md                   Project documentation (this file)
```

---

## Dataset

- **Source:** [Yahoo Finance — TSLA Historical Data](https://finance.yahoo.com/quote/TSLA/history)
- **Ticker:** TSLA (NASDAQ)
- **Coverage:** June 29, 2010 (IPO) → December 29, 2023
- **Frequency:** Daily (trading days only, ~252/year)
- **Total Rows:** 3,400

### Columns

| Column | Description | Used in Model |
|---|---|---|
| Date | Trading date | Yes — as temporal index |
| Open | First trade price of the day | No |
| High | Intraday maximum price | No |
| Low | Intraday minimum price | No |
| **Close** | **Last trade price of the day** | **Yes — target variable** |
| Adj Close | Dividend/split-adjusted close | No |
| Volume | Total shares traded | No |

> **Design Decision:** Only the `Close` price is used. This is an intentional **univariate autoregressive** design — the model learns to predict tomorrow's closing price from the previous 3 closing prices alone, without any auxiliary features. This isolates the LSTM's core temporal learning capability.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TESLA STOCK PREDICTION SYSTEM                        │
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────┐  │
│  │  RAW     │    │    DATA      │    │ DEEP LEARNING │    │ OUTPUT  │  │
│  │  DATA    │───▶│  PIPELINE    │───▶│    ENGINE     │───▶│  LAYER  │  │
│  │  LAYER   │    │    LAYER     │    │    LAYER      │    │         │  │
│  └──────────┘    └──────────────┘    └───────────────┘    └─────────┘  │
│   TSLA.csv        Preprocessing       LSTM Model          Predictions  │
│   3400 rows       Windowing           Training            Recursive    │
│   7 columns       Splitting           Evaluation          Export       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Architecture

```
                     ┌─────────────────────┐
                     │      TSLA.csv       │
                     │  3400 rows × 7 cols │
                     │  2010-06-29 to      │
                     │  2023-12-29         │
                     └──────────┬──────────┘
                                │
                                ▼
                   ┌────────────────────────────┐
                   │     FEATURE SELECTION      │
                   │  Keep: Date, Close only    │
                   │  Drop: Open, High, Low,    │
                   │        Adj Close, Volume   │
                   │  3400 rows × 2 cols        │
                   └────────────┬───────────────┘
                                │
                                ▼
                   ┌────────────────────────────┐
                   │    DATE TRANSFORMATION     │
                   │  "2010-06-29" (string)     │
                   │           ───▶             │
                   │  datetime(2010, 6, 29)     │
                   │  Set as DataFrame Index    │
                   └────────────┬───────────────┘
                                │
                                ▼
                   ┌────────────────────────────┐
                   │   SLIDING WINDOW ENGINE    │
                   │   n = 3 day lookback       │
                   │                            │
                   │  For each trading day:     │
                   │  [day-3, day-2, day-1]     │
                   │          ───▶              │
                   │        [day target]        │
                   │                            │
                   │  Calendar-aware:           │
                   │  skips weekends/holidays   │
                   │  3400 → 3397 rows          │
                   └────────────┬───────────────┘
                                │
                                ▼
                   ┌────────────────────────────┐
                   │      ARRAY RESHAPING       │
                   │                            │
                   │  dates → (3397,)           │
                   │  X     → (3397, 3, 1)      │
                   │  y     → (3397,)           │
                   │                            │
                   │  [samples, timesteps,      │
                   │   features] for LSTM       │
                   └────────────┬───────────────┘
                                │
                   ┌────────────┼────────────┐
                   │            │            │
                   ▼            ▼            ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐
            │  TRAIN   │ │   VAL    │ │   TEST   │
            │   80%    │ │   10%    │ │   10%    │
            │2717 rows │ │ 340 rows │ │ 340 rows │
            │2010─2021 │ │2021─2022 │ │2022─2023 │
            └────┬─────┘ └────┬─────┘ └────┬─────┘
                 └────────────┼────────────┘
                              │
                              ▼
                   ┌────────────────────────────┐
                   │       LSTM MODEL           │
                   │  Input(3,1) → LSTM(64)     │
                   │  → Dense(32) → Dense(32)   │
                   │  → Dense(1)                │
                   │  Adam(lr=0.001), MSE loss  │
                   │  100 epochs, batch=32      │
                   └────────────┬───────────────┘
                                │
                     ┌──────────┴──────────┐
                     │                     │
                     ▼                     ▼
          ┌─────────────────┐   ┌──────────────────────┐
          │   STANDARD      │   │   RECURSIVE          │
          │   PREDICTIONS   │   │   FORECAST           │
          │                 │   │                      │
          │  Feed real X    │   │  Feed model's own    │
          │  → get ŷ        │   │  output back as      │
          │  train/val/test │   │  input — 14 steps    │
          └────────┬────────┘   └──────────┬───────────┘
                   └──────────┬────────────┘
                              │
                              ▼
                   ┌────────────────────────────┐
                   │      VISUALIZATION         │
                   │  All splits + recursive    │
                   │  Predictions vs Actuals    │
                   └────────────┬───────────────┘
                                │
                                ▼
                   ┌────────────────────────────┐
                   │      MODEL EXPORT          │
                   │      my_model.keras        │
                   │  ZIP: arch + weights +     │
                   │       optimizer state      │
                   └────────────────────────────┘
```

---

## LSTM Model Architecture

```
INPUT SEQUENCE — 3 trading days × 1 price value
┌──────────────────────────────────────────────────┐
│  Timestep 1      Timestep 2      Timestep 3      │
│  [day t-3]       [day t-2]       [day t-1]       │
│  e.g: 239.37     242.64          243.84           │
└──────────────────────────────────────────────────┘
                        │
                        │  Shape: (batch, 3, 1)
                        ▼
┌──────────────────────────────────────────────────┐
│               Input Layer                        │
│               shape = (3, 1)                     │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│                 LSTM Layer                       │
│                 64 units                         │
│                                                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐    │
│  │  Cell 1   │─▶│  Cell 2   │─▶│  Cell 3   │    │
│  │ timestep 1│  │ timestep 2│  │ timestep 3│    │
│  └───────────┘  └───────────┘  └─────┬─────┘    │
│                                      │           │
│  Each cell has:                      │           │
│  • Forget gate  — erase old memory   │           │
│  • Input gate   — add new memory     │           │
│  • Cell state   — long-term memory   │           │
│  • Output gate  — what to pass on    │           │
│                                      │           │
│  Output: final hidden state only     │           │
│  Shape:  (batch, 64)         ────────┘           │
│  Parameters: 16,896                              │
└───────────────────────┬──────────────────────────┘
                        │
                        │  Shape: (batch, 64)
                        ▼
┌──────────────────────────────────────────────────┐
│              Dense Layer 1                       │
│              32 neurons  |  ReLU activation      │
│              f(x) = max(0, x)                    │
│              Parameters: 64×32 + 32 = 2,080      │
└───────────────────────┬──────────────────────────┘
                        │
                        │  Shape: (batch, 32)
                        ▼
┌──────────────────────────────────────────────────┐
│              Dense Layer 2                       │
│              32 neurons  |  ReLU activation      │
│              f(x) = max(0, x)                    │
│              Parameters: 32×32 + 32 = 1,056      │
└───────────────────────┬──────────────────────────┘
                        │
                        │  Shape: (batch, 32)
                        ▼
┌──────────────────────────────────────────────────┐
│              Output Layer                        │
│              1 neuron  |  Linear (no activation) │
│              Parameters: 32×1 + 1 = 33           │
└───────────────────────┬──────────────────────────┘
                        │
                        │  Shape: (batch, 1)
                        ▼
               ┌──────────────────┐
               │  PREDICTED       │
               │  CLOSING PRICE   │
               │  e.g: $237.33    │
               └──────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL PARAMETERS:   20,065   (~78 KB model size)
Trainable:          20,065
Non-trainable:           0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## LSTM Cell — Internal Mechanics

```
One LSTM Cell (unrolled across 3 timesteps):

   Previous Hidden State  h(t-1)
   Previous Cell State    c(t-1)
              │
              ▼
 x(t) ──▶  ┌──────────────────────────────────────────────┐
            │                 LSTM CELL                    │
            │                                              │
            │  ┌──────────┐                                │
            │  │ sigmoid  │  Forget Gate                   │
            │  │    ft    │◀── concat[h(t-1), x(t)] + bias │
            │  └────┬─────┘  "What % of c(t-1) to keep?"  │
            │       │   ft ∈ [0, 1]                        │
            │  ┌────▼─────┐  ┌──────────┐                  │
            │  │     ×    │  │ sigmoid  │  Input Gate      │
            │  └────┬─────┘  │    it    │                  │
            │       │        └────┬─────┘                  │
            │       │        ┌────▼─────┐                  │
            │  ┌────▼─────┐  │  tanh    │  Candidate State │
            │  │     +    │◀─│   g̃(t)  │                  │
            │  └────┬─────┘  └──────────┘                  │
            │       │                                      │
            │   c(t) = ft ⊗ c(t-1)  +  it ⊗ g̃(t)         │
            │   (updated long-term cell state)             │
            │       │                                      │
            │  ┌────▼─────┐  ┌──────────┐                  │
            │  │  tanh     │  │ sigmoid  │  Output Gate    │
            │  └────┬─────┘  │    ot    │                  │
            │       │        └────┬─────┘                  │
            │  ┌────▼─────┐       │                        │
            │  │     ×    │◀──────┘                        │
            │  └────┬─────┘                                │
            └───────┼──────────────────────────────────────┘
                    │
             h(t) = ot ⊗ tanh(c(t))
             (output hidden state passed to next cell and Dense layers)
                    │
           ┌────────┴────────┐
           ▼                 ▼
    Next Cell State    Next Timestep
       c(t) →              h(t) →
```

---

## Windowing Architecture

```
RAW TIME SERIES (Close prices, daily):
Index:   0        1        2        3        4        5   ...
Date:  Jun29    Jun30    Jul01    Jul02    Jul06    Jul07  ...
Price: 1.593    1.589    1.464    1.280    1.074    1.053  ...

SLIDING WINDOW (n=3, step=1 trading day):

Window 1: [1.593, 1.589, 1.464]  →  TARGET: 1.280   (Jul02)
           t-3     t-2    t-1                   t

Window 2: [1.589, 1.464, 1.280]  →  TARGET: 1.074   (Jul06)
Window 3: [1.464, 1.280, 1.074]  →  TARGET: 1.053   (Jul07)
Window 4: [1.280, 1.074, 1.053]  →  TARGET: 1.164   (Jul08)
  ...
Window 3397: [261.44, 253.18, 248.48]  →  TARGET: ?  (Dec29)

CALENDAR-AWARE NEXT-DAY LOGIC:
┌─────────────────────────────────────────────────────────┐
│  target_date                                            │
│       │                                                 │
│       ▼                                                 │
│  Look ahead 7 calendar days                             │
│       │                                                 │
│       ▼                                                 │
│  .head(2).tail(1)  →  picks the 2nd row = next          │
│                       actual trading day                │
│       │                                                 │
│       ▼                                                 │
│  Parse numpy datetime64 string:                         │
│  "2023-12-26T00:00:00.000000000"                        │
│       │  .split('T')[0]                                 │
│       ▼                                                 │
│  "2023-12-26"  →  datetime(2023, 12, 26)                │
│                                                         │
│  Automatically skips: Saturdays, Sundays, US holidays   │
└─────────────────────────────────────────────────────────┘

FINAL WINDOWED DATAFRAME (3397 rows × 5 cols):
┌──────────────┬──────────┬──────────┬──────────┬────────────┐
│ Target Date  │ Target-3 │ Target-2 │ Target-1 │   Target   │
├──────────────┼──────────┼──────────┼──────────┼────────────┤
│ 2010-07-02   │  1.5927  │  1.5887  │  1.4640  │   1.2800   │
│ 2010-07-06   │  1.5887  │  1.4640  │  1.2800  │   1.0740   │
│    ...       │   ...    │   ...    │   ...    │    ...     │
│ 2023-12-29   │ 261.44   │ 253.18   │ 248.48   │    ...     │
└──────────────┴──────────┴──────────┴──────────┴────────────┘
```

---

## Train / Validation / Test Split

```
FULL DATASET: 3397 samples  (2010-07-02 → 2023-12-29)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

│◄──────────────── TRAIN 80% ─────────────────►│◄─ VAL 10% ─►│◄─ TEST 10% ─►│
│               2717 samples                   │  340 samp.  │  340 samp.   │
│            2010-07-02 → ~2021                │  2021─2022  │  2022─2023   │
│                                              │             │              │
0                                            2717          3057           3397

TEMPORAL INTEGRITY RULES:
  Train  → model LEARNS from this
  Val    → monitor overfitting during training (no weight update)
  Test   → completely hidden until final evaluation
           model has ZERO knowledge of these prices

WHY NOT RANDOM SHUFFLE?
  If shuffled: model trains on 2023 data and tests on 2020 data
  That is DATA LEAKAGE — the model "sees the future" during training
  Temporal split mirrors real-world deployment: always predict forward
```

---

## Training Loop Architecture

```
EPOCH  (repeated 100 times):

  X_train shape: (2717, 3, 1)
       │
       │  mini-batches: 32 samples each  →  85 batches per epoch
       ▼
┌───────────────────────────────────────────────────────┐
│   FORWARD PASS                                        │
│   Input ──▶ LSTM(64) ──▶ Dense(32) ──▶ Dense(32)     │
│          ──▶ Dense(1)  ──▶  predicted price ŷ         │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│   LOSS CALCULATION                                    │
│   MSE = (1/n) × Σ (y_actual − y_pred)²               │
│   MAE = (1/n) × Σ |y_actual − y_pred|  (displayed)   │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│   BACKWARD PASS  (Backpropagation Through Time)       │
│   Compute ∂Loss/∂weights for all layers               │
│   Unroll LSTM 3 steps back through time               │
│   Propagate gradients: Output → Dense → LSTM          │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│   ADAM OPTIMIZER  (lr = 0.001)                        │
│                                                       │
│   m_t = β1·m(t-1) + (1−β1)·gradient    β1=0.9        │
│   v_t = β2·v(t-1) + (1−β2)·gradient²   β2=0.999      │
│   m̂_t = m_t  / (1 − β1^t)   ← bias correction        │
│   v̂_t = v_t  / (1 − β2^t)   ← bias correction        │
│   w   = w − lr × m̂_t / (√v̂_t + ε)    ε=1e-7         │
└───────────────────────┬───────────────────────────────┘
                        │
                   85 batches done
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│   VALIDATION CHECK  (end of each epoch)               │
│   Run X_val through model — no weight updates         │
│   Report: val_loss, val_mae                           │
│   Detect overfitting: train_loss↓ + val_loss↑ = BAD  │
└───────────────────────────────────────────────────────┘
                  (repeat 100 epochs)
```

---

## Recursive Forecasting Architecture

```
STANDARD PREDICTION — uses real historical data each step:

  Input (real):  [239.37, 242.64, 243.84]  ──▶  MODEL  ──▶  ŷ₁
  Input (real):  [242.64, 243.84, real₃]   ──▶  MODEL  ──▶  ŷ₂
  (ground-truth values always available as input)


RECURSIVE PREDICTION — self-feeding, no real future data needed:

  Step 1:  [239.37, 242.64, 243.84]  ──▶  MODEL  ──▶  237.33
                                                           │
  Step 2:  [242.64, 243.84, 237.33]  ──▶  MODEL  ──▶  233.26
                                                           │
  Step 3:  [243.84, 237.33, 233.26]  ──▶  MODEL  ──▶  228.85
                                                           │
  Step 4:  [237.33, 233.26, 228.85]  ──▶  MODEL  ──▶  225.25
  ...continues 14 steps...

  Window mechanism each step:
    new_window = last_window[1:]         drop oldest value (shift left)
    new_window.append(next_prediction)   append model's own output
    last_window = np.array(new_window)   update for next iteration

OBSERVED BEHAVIOUR:
  Predictions converge → $217 range (model finds attractor state)
  This reveals: without real data feedback, the model drifts
  This is both a strength (stable) and a limitation (can't surprise itself)
```

---

## Parameter Count Breakdown

```
┌───────────────────────────────────────────────────────────────┐
│                    PARAMETER BREAKDOWN                        │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  LSTM Layer (64 units, input_dim=1):                          │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Input weights:     4 × (64 × 1)  =       256          │  │
│  │  Recurrent weights: 4 × (64 × 64) =    16,384          │  │
│  │  Biases:            4 × 64        =       256          │  │
│  │                                   ───────────          │  │
│  │  Subtotal:                         16,896              │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  Dense Layer 1  (64 → 32, ReLU):                              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Weights: 64 × 32 = 2,048                               │  │
│  │  Biases:       32 =    32                               │  │
│  │                   ───────────                           │  │
│  │  Subtotal:          2,080                               │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  Dense Layer 2  (32 → 32, ReLU):                              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Weights: 32 × 32 = 1,024                               │  │
│  │  Biases:       32 =    32                               │  │
│  │                   ───────────                           │  │
│  │  Subtotal:          1,056                               │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  Output Layer  (32 → 1, Linear):                              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Weights: 32 × 1 =   32                                 │  │
│  │  Biases:       1 =    1                                 │  │
│  │                  ─────────                              │  │
│  │  Subtotal:           33                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  GRAND TOTAL:  16,896 + 2,080 + 1,056 + 33  =  20,065        │
│  Model file size on disk: ~78 KB                              │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
└───────────────────────────────────────────────────────────────┘
```

---

## Stage-by-Stage Code Walkthrough

### Stage 1 — Import Libraries

```python
import datetime
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
```

| Import | Purpose |
|---|---|
| `datetime` | Parse date strings; compute next trading day via `timedelta` |
| `keras` | Top-level API for `.keras` format model save/load |
| `matplotlib.pyplot` | All 4 visualizations: raw price, splits, predictions, recursive |
| `numpy` | Array reshaping, float32 conversion, LSTM input matrix |
| `pandas` | CSV loading, column selection, DatetimeIndex operations |
| `Sequential` | Linear layer stack — defines the model graph |
| `Adam` | Adaptive optimizer with per-parameter learning rates |
| `layers` | LSTM, Dense layer classes |

---

### Stage 2 — Load Dataset

```python
df = pd.read_csv('TSLA.csv')
```

Loads 3,400 rows × 7 columns. The Date column is a plain string at this point (`object` dtype). All 7 OHLCV columns are loaded but most will be dropped.

---

### Stage 3 — Feature Selection

```python
df = df[['Date', 'Close']]
```

Reduces to 2 columns: `Date` (string) and `Close` (float64). Dropping High/Low/Volume removes collinear features and noise for this univariate experiment.

---

### Stage 4 — Date Parsing and Indexing

```python
def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

df['Date'] = df['Date'].apply(str_to_datetime)
df.index = df.pop('Date')
```

- Custom parser (not `pd.to_datetime`) because the same function is reused inside the windowing loop to parse numpy `datetime64` string representations like `"2023-12-26T00:00:00.000000000"`.
- `df.pop('Date')` removes the column and returns it — set as DataFrame index.
- Now `df.loc[:target_date]` and `df.loc[date1:date2]` work as temporal slices.

---

### Stage 5 — Windowed DataFrame

```python
windowed_df = df_to_windowed_df(df, '2010-07-02', '2023-12-29', n=3)
```

The core preprocessing function. For each trading day, collects the previous `n=3` closing prices as features and the current price as the label. The next-trading-day logic is calendar-aware — it looks up to 7 days ahead and picks the first real market day, automatically skipping weekends and holidays.

**Result:** 3,397 rows × 5 columns (`Target Date, Target-3, Target-2, Target-1, Target`).

---

### Stage 6 — Reshape for LSTM

```python
dates, X, y = windowed_df_to_date_X_y(windowed_df)
# X.shape → (3397, 3, 1)   [samples, timesteps, features]
```

LSTM requires a 3D input tensor `(samples, timesteps, features)`. The `middle_matrix` of shape `(3397, 3)` is reshaped to `(3397, 3, 1)` by adding a `features=1` dimension. Converted to `float32` for TensorFlow compatibility.

---

### Stage 7 — Train/Val/Test Split

```python
q_80 = int(len(dates) * .8)  # 2717
q_90 = int(len(dates) * .9)  # 3057
```

Strict chronological split. No shuffling. Train → Val → Test proceeds forward in time, exactly as a deployed model would encounter new data.

---

### Stage 8 — Build LSTM Model

```python
model = Sequential([
    layers.Input((3, 1)),
    layers.LSTM(64),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
```

- `Input((3, 1))` — explicit input shape for a fully-specified computation graph before compilation
- `LSTM(64)` — 64 memory units; processes 3 timesteps, outputs 64-dimensional temporal encoding
- `Dense(32, relu)` × 2 — translates the temporal encoding to a scalar prediction through two non-linear transformations
- `Dense(1)` — linear output (no activation) for unbounded regression output

---

### Stage 9 — Compile and Train

```python
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
```

- **Loss: MSE** — penalises large price errors more than small ones (quadratic penalty)
- **Optimizer: Adam** — auto-adjusts learning rates per parameter; handles noisy financial gradients without manual tuning
- **MAE metric** — human-readable error in dollars, shown each epoch
- **`validation_data`** — monitors generalization each epoch without updating weights

---

### Stage 10 — Evaluate on All Three Sets

```python
train_predictions = model.predict(X_train).flatten()
val_predictions   = model.predict(X_val).flatten()
test_predictions  = model.predict(X_test).flatten()
```

`.flatten()` converts `(n, 1)` → `(n,)` for direct plotting. Each set is plotted independently, then combined on a single chart showing the full 13-year timeline with all 6 series.

---

### Stage 11 — Recursive Forecast

```python
for target_date in recursive_dates:
    next_prediction = model.predict(np.array([last_window])).flatten()
    recursive_predictions.append(next_prediction)
    new_window = list(last_window[1:])
    new_window.append(next_prediction)
    last_window = np.array(new_window)
```

14-step self-feeding forecast. Each prediction replaces the oldest value in the window — simulating forward prediction with no ground-truth data available.

---

### Stage 12 — Export Model

```python
model.save("my_model.keras")
reconstructed_model = keras.models.load_model("my_model.keras")
```

Saves as a ZIP archive containing architecture JSON, weight tensors, and optimizer state. The `.keras` format (Keras 3 native) supersedes the deprecated `.h5` (HDF5) format.

---

## Technologies Used

| Technology | Version | Role |
|---|---|---|
| Python | 3.10+ | Core language |
| TensorFlow | 2.17.0 | GPU-accelerated deep learning backend |
| Keras | 3.5.0 | High-level neural network API |
| NumPy | 2.1.1 | N-dimensional array math and matrix operations |
| Pandas | 2.2.3 | Tabular data manipulation, DatetimeIndex |
| Matplotlib | 3.9.2 | All visualizations and plots |
| Jupyter Notebook | — | Interactive development environment |

---

## How to Run

### 1 — Clone the Repository

```bash
git clone https://github.com/sunkireddy-Barath/tesla-stock-prediction-lstm.git
cd tesla-stock-prediction-lstm
```

### 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` pins exact versions for full reproducibility:

```
keras==3.5.0
matplotlib==3.9.2
numpy==2.1.1
pandas==2.2.3
tensorflow==2.17.0
```

### 3 — Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `TSLA_Stock_LSTM.ipynb` and run all cells sequentially from top to bottom.

### 4 — View Pre-rendered Output (no code required)

Open `TSLA_Stock_LSTM.html` in any web browser to see the full notebook output including all plots — without installing anything.

---

## Results and Observations

### Model Performance Summary

| Prediction Set | Behaviour |
|---|---|
| Training (80%) | Closely tracks the actual price curve across all 13 years |
| Validation (10%) | Maintains prediction quality on unseen 2021–2022 data |
| Test (10%) | Follows major trends in the 2022–2023 period |
| Recursive (14 steps) | Converges toward ~$217 — model finds a stable attractor |

### Key Observations

1. **Short-term accuracy** — The model accurately captures day-to-day price momentum for the training and validation periods.
2. **Trend following** — LSTM successfully identifies major bull runs (2020–2021) and corrections (2022).
3. **Recursive drift** — The 14-step recursive forecast underestimates recovery, converging to a stable value rather than tracking the actual upswing. This is expected and demonstrates the inherent limitation of pure autoregressive forecasting without external signals.
4. **No scaling required** — The model converges without MinMaxScaler normalization, demonstrating the robustness of the Adam optimizer to handle wide price ranges ($1.59 to $409+).

---

## Why This Approach is Different

| Aspect | Typical Tutorial | This Project |
|---|---|---|
| Windowing | Naive numpy slicing, ignores date gaps | Calendar-aware: finds next actual trading day |
| Date handling | Usually dropped or integer indices | Preserved throughout — enables temporal visualization |
| Data split | Often random shuffle | Strictly temporal — no data leakage |
| Prediction types | Only standard predictions | Both standard AND recursive (14-step self-feeding forecast) |
| Model export | Not demonstrated | Full save/load cycle with architecture verification |
| Validation | Train/Test 2-split | Train/Validation/Test 3-split (proper held-out evaluation) |
| Input shape | Often implicit | Explicit Input layer — fully specified computation graph |
| Date parsing | `pd.to_datetime()` only | Custom parser reusable across both CSV and numpy datetime64 |

---

## Limitations

- **Univariate only** — The model uses only closing price history. Real-world price movements are driven by earnings reports, macroeconomic data, news events, and market sentiment — none of which are captured here.
- **Short window (n=3)** — Three days of history captures only immediate momentum. Monthly and quarterly patterns require windows of 30–60 days.
- **No normalization** — Training on raw dollar values (range: $1 to $400+) instead of normalized [0,1] values may reduce training stability for larger price ranges.
- **No early stopping** — Training runs for the full 100 epochs regardless of validation loss convergence.
- **Recursive forecast divergence** — Self-feeding predictions accumulate error each step; beyond ~7 steps the forecast loses fidelity.
- **Not suitable for live trading** — This is a research/educational model. It should not be used for real financial decisions.

---

## Future Enhancements

- Add `MinMaxScaler` normalization with inverse transform for predictions
- Increase window size to `n=30` or `n=60` for monthly pattern capture
- Add technical indicators as features: RSI, MACD, Bollinger Bands, EMA
- Implement stacked LSTM layers with `return_sequences=True`
- Add `EarlyStopping` callback with `restore_best_weights=True`
- Compare with GRU (Gated Recurrent Unit) — faster training, similar accuracy
- Add Bidirectional LSTM for non-real-time retrospective analysis
- Integrate attention mechanism to weight which past days matter most
- Add Twitter/Reddit sentiment as an external input feature
- Deploy as a web application with live Yahoo Finance data feed
- Add ARIMA and Prophet baselines for comparative benchmarking

---

## Conclusion

This project demonstrates a complete, production-structured LSTM pipeline for financial time-series forecasting. It implements calendar-aware windowing, strict temporal data splitting, three-phase evaluation (train/validation/test), and recursive multi-step forecasting — going well beyond a standard tutorial.

The 20,065-parameter LSTM model, trained on 13 years of Tesla trading history, successfully captures the major temporal patterns in a highly volatile stock — including the exponential growth phase (2019–2021), the peak correction (2022), and partial recovery (2023) — without any feature normalization, which demonstrates both the expressiveness of the LSTM architecture and the convergence stability of the Adam optimizer.

The recursive forecasting section specifically illustrates the distinction between **backtesting** (evaluating on known historical data) and **true forecasting** (feeding model output as future input) — a critical distinction in quantitative finance that separates research-grade work from production deployment.

---

<div align="center">

**Built by [Sunkireddy Barath](https://github.com/sunkireddy-Barath)**

Dataset: [Yahoo Finance — TSLA](https://finance.yahoo.com/quote/TSLA) &nbsp;|&nbsp;
Framework: [TensorFlow](https://tensorflow.org) + [Keras](https://keras.io) &nbsp;|&nbsp;
Language: Python 3.10+

</div>
