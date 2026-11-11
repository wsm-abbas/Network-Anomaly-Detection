# Network Anomaly Detection

Models that spot unusual network activity, with a focus on **false alarms**.
In a real security setting a detector that cries wolf all day gets ignored, so
besides precision and recall I also look at how much normal traffic gets
flagged by mistake.

I compare two approaches:

- **XGBoost** (supervised) - learns from labelled attacks.
- **Isolation Forest** (unsupervised) - learns what normal looks like and flags
  anything that doesn't fit.

## Tools

Python, pandas, scikit-learn, XGBoost.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

```bash
python src/make_data.py        # creates data/traffic.csv
python src/run_models.py       # trains and compares the models
```

Run the second command from the project folder (the one with `data/` in it).

## About the data

`src/make_data.py` generates fake traffic with about 4% attacks (port scans,
floods, exfiltration). It is only there so the project runs out of the box.
To use a real dataset such as NSL-KDD, convert it to numeric features and save
it as `data/traffic.csv` with a `label` column (0 normal, 1 attack).

## Results

30% held-out test set (6,000 records, 240 attacks). Also in
`results/comparison.csv`; trained models are in `models/`.

| Model | Precision | Recall | F1 | False alarm rate |
|-------|-----------|--------|----|------------------|
| XGBoost | 0.909 | 0.963 | 0.935 | 0.40% |
| XGBoost (cutoff 0.8) | 0.933 | 0.929 | 0.931 | 0.28% |
| Isolation Forest | 0.520 | 0.963 | 0.675 | 3.70% |

Isolation Forest catches as many attacks as XGBoost but raises about 9 times
as many false alarms, because it flags anything unusual, including big but
harmless transfers. XGBoost has seen labelled attacks so it is far more
selective. The data is generated, so treat this as a demo, not a finding.

## Notes

- Accuracy is not reported on purpose. With 96% normal traffic, a model that
  never flags anything already scores 96%.
- Moving the XGBoost cutoff is a quick way to trade missed attacks for fewer
  false alarms.

## Things I would try next

- Time-based split instead of a random one.
- Test on a different dataset to see how it copes with unfamiliar traffic.
# Network-Anomaly-Detection
