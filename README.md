# Industrial Sensor Anomaly Detection — PyTorch LSTM Autoencoder

A deep learning project that uses a PyTorch LSTM Autoencoder to detect anomalies in industrial sensor time-series data, simulating a real-world predictive maintenance system for industrial automation companies like ABB.

The project includes two implementations:
1. **Synthetic sensor data** — proof of concept using generated temperature, vibration, and pressure readings with manually injected anomalies
2. **Real NASA SMAP telemetry data** — production-grade anomaly detection on real spacecraft sensor data with NASA-provided ground truth labels

---

## Project Overview

Predictive maintenance is one of the most critical applications of AI in industrial automation — detecting abnormal sensor behavior before it leads to equipment failure saves companies millions in unplanned downtime. This project builds an LSTM Autoencoder that learns what "normal" sensor patterns look like, then flags any sequence that deviates significantly as an anomaly.

The core idea: train the model only on normal data, so it learns to reconstruct normal patterns well. When it encounters an anomaly, reconstruction error spikes — and that spike is the anomaly signal.

---

## Datasets

### 1. Synthetic Industrial Sensor Data (Proof of Concept)
- 1,000 timesteps of simulated temperature, vibration, and pressure readings
- Anomalies manually injected at timesteps: 200, 201, 202, 400, 401, 600, 601, 602, 603, 800
- Used to validate the LSTM Autoencoder architecture before applying to real data

### 2. NASA SMAP Telemetry Dataset (Real World)
- Real spacecraft telemetry data published by NASA's Jet Propulsion Laboratory
- Channel used: **P-1** (pressure/flow sensor)
- Training samples: 2,872 timesteps (25 channels)
- Test samples: 8,505 timesteps (25 channels)
- Ground truth anomaly zones (NASA-labeled):
  - Timesteps 2149–2349
  - Timesteps 3539–3779
  - Timesteps 4536–4844
- Source: [NASA SMAP/MSL Dataset](https://github.com/khundman/telemanom)

---

## Model Architecture

A custom LSTM Autoencoder built with PyTorch:

```
Input (sequence_length x features)
 → Encoder: LSTM(input_size, hidden=64, layers=2)
 → Context vector (last hidden state repeated)
 → Decoder: LSTM(hidden=64, output=input_size, layers=2)
 → Reconstructed sequence
```

- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam (lr=0.001)
- **Anomaly threshold:** Mean + 2 × Std of training reconstruction errors
- **Anomaly signal:** Reconstruction error exceeding threshold = anomaly flagged

The model is trained only on normal data. At inference time, sequences with reconstruction error above the threshold are flagged as anomalies.

---

## Results

### Synthetic Data
| Metric | Value |
|---|---|
| Training epochs | 50 |
| Final training loss | 0.083545 |
| Anomaly threshold | 0.1406 |
| Anomalies detected | 86 sequences |
| Injected anomaly clusters | 4 clusters — all detected ✅ |

### Real NASA SMAP P-1 Data
| Metric | Value |
|---|---|
| Training epochs | 50 |
| Final training loss | 0.041499 |
| Anomaly threshold | 0.095544 |
| Total sequences tested | 8,475 |
| Anomalies detected | 523 sequences |
| NASA-labeled anomaly zones | 3 zones — all detected ✅ |

The model's reconstruction error spikes align precisely with all three NASA-labeled anomaly zones (2149–2349, 3539–3779, 4536–4844), confirming the model correctly learned normal sensor patterns and successfully identifies real deviations.

---

## Key Technical Decisions

**Why LSTM Autoencoder?**
Time-series sensor data has temporal dependencies — what happens at timestep T depends on what happened at T-1, T-2, etc. LSTMs are specifically designed to capture these dependencies. The autoencoder framework allows unsupervised anomaly detection — no labeled anomaly data needed during training, which matches real industrial scenarios where failures are rare and hard to label in advance.

**Why reconstruction error as anomaly score?**
The model is trained only on normal data, so it learns to reconstruct normal patterns well. Anomalous sequences, by definition, don't match what the model learned — reconstruction error spikes. This is a well-established technique in industrial anomaly detection.

**Why mean + 2 std as threshold?**
This is a statistically principled choice — it flags sequences that deviate more than 2 standard deviations from the average training reconstruction error. In production, this threshold would be tuned based on the acceptable false positive rate for the specific use case.

---

## Project Structure

```
PyTorch-Anomaly-Detection/
│
├── main.py                        # Synthetic sensor data — proof of concept
├── real_anomaly_detection.py      # Real NASA SMAP P-1 anomaly detection
├── data/                          # NASA SMAP dataset (gitignored)
│   └── archive/
│       ├── labeled_anomalies.csv  # Ground truth anomaly labels
│       └── data/
│           ├── train/             # .npy training files per channel
│           └── test/              # .npy test files per channel
├── abb_anomaly_model.pt           # Saved synthetic data model
├── abb_real_anomaly_model.pt      # Saved real data model
├── anomaly_detection_results.png  # Synthetic data results plot
├── real_anomaly_results.png       # NASA SMAP results plot
├── requirements.txt
└── README.md
```

---

## Output Graphs

Each run produces a 4-panel visualization:

1. **Raw sensor signal** — the actual time-series data being analyzed
2. **Training loss curve** — confirms the model converged properly
3. **Reconstruction error vs real labels** — orange zones mark NASA-labeled anomaly regions; reconstruction error spikes align with them
4. **Flagged anomalies** — red dots mark every detected anomaly point above the threshold

---

## How to Run

1. **Set up environment**
```bash
python -m venv pytorch_env
pytorch_env\Scripts\activate      # Windows
pip install torch torchvision torchaudio scikit-learn pandas numpy matplotlib seaborn tensorboard
```

2. **Run synthetic data proof of concept**
```bash
python main.py
```

3. **Run real NASA SMAP anomaly detection**
   - Download the NASA SMAP dataset from [Kaggle](https://www.kaggle.com/) or the [original source](https://github.com/khundman/telemanom)
   - Extract into `data/archive/`
   - Run:
```bash
python real_anomaly_detection.py
```

---

## Tech Stack

- Python 3.11
- PyTorch
- NumPy, Pandas
- Scikit-learn (preprocessing)
- Matplotlib (visualization)

---

## Relevance to Industrial Automation

ABB's industrial systems generate continuous streams of sensor data from motors, drives, robots, and power equipment. Anomaly detection on this data is a core component of predictive maintenance pipelines — flagging abnormal behavior before it becomes a failure. This project demonstrates the full pipeline: data ingestion, sequence modeling with LSTMs, threshold-based anomaly scoring, and visualization of results against ground truth labels — the same workflow used in production industrial AI systems.

The approach generalizes directly to ABB equipment: replace the NASA P-1 telemetry channel with readings from ABB motor drives, robot joints, or power transformers, and the same LSTM Autoencoder pipeline applies.

---

## Author

Yash Gupta
GitHub: [YashGupta018](https://github.com/YashGupta018)
