# Federated Continual Learning for Financial Fraud Detection

A capstone project implementing:

* Preprocessing with leakage prevention
* Time-based train/validation/test splits
* Continual Learning with Replay Buffers
* Federated Learning across multiple banks
* Federated Continual Learning (FCL)

This project simulates multiple banks collaboratively detecting fraud **without sharing raw customer data**.

---

# Project Structure

```text
final_capstone/
│── data/
│   ├── Bank_A.csv
│   ├── Bank_B.csv
│   └── Bank_C.csv
│
│── processed/
│   ├── bankA.pkl
│   ├── bankB.pkl
│   └── bankC.pkl
│
│── continual/
│   └── bank_train.py
│
│── federated/
│   ├── server.py
│   ├── client.py
│   └── client_fcl.py
│
│── preprocess.py
│── results_report.py
│── requirements.txt
│── README.md
```

---

# Setup Instructions

## 1. Create Virtual Environment

### Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

# requirements.txt

Create a file named `requirements.txt` with:

```txt
flwr
torch
pandas
numpy
scikit-learn
joblib
scipy
```

---

# Add Dataset Files

Place these CSV files inside the `data/` folder:

```text
Bank_A.csv
Bank_B.csv
Bank_C.csv
```

---

# Run the Project

# Step 1: Preprocess All Banks

This will:

* remove leakage columns
* handle missing values
* scale features
* encode categorical values
* create time-based splits
* save processed `.pkl` files

Run:

```bash
python3 preprocess.py
```

Expected output:

```text
processed/
├── bankA.pkl
├── bankB.pkl
└── bankC.pkl
```

---

# Step 2: Run Local Continual Learning

This trains one bank locally using:

* sequential tasks
* replay buffer
* fraud detection model

## Run Bank A

```bash
python3 continual/bank_train.py
```

## Run Bank B

Open `continual/bank_train.py` and change:

```python
run_bank("processed/bankA.pkl")
```

to:

```python
run_bank("processed/bankB.pkl")
```

Then run:

```bash
python3 continual/bank_train.py
```

## Run Bank C

Change to:

```python
run_bank("processed/bankC.pkl")
```

Then run:

```bash
python3 continual/bank_train.py
```

---

# Step 3: Run Federated Learning (Baseline)

Open **4 terminals**.

## Terminal 1 — Start Server

```bash
python3 federated/server.py
```

## Terminal 2 — Bank A Client

```bash
python3 federated/client.py processed/bankA.pkl
```

## Terminal 3 — Bank B Client

```bash
python3 federated/client.py processed/bankB.pkl
```

## Terminal 4 — Bank C Client

```bash
python3 federated/client.py processed/bankC.pkl
```

---

# Step 4: Run Federated Continual Learning (Final Model)

Open **4 terminals**.

## Terminal 1 — Start Server

```bash
python3 federated/server.py
```

## Terminal 2 — Bank A

```bash
python3 federated/client_fcl.py processed/bankA.pkl
```

## Terminal 3 — Bank B

```bash
python3 federated/client_fcl.py processed/bankB.pkl
```

## Terminal 4 — Bank C

```bash
python3 federated/client_fcl.py processed/bankC.pkl
```

---

# Step 5: Generate Results Report

This prints:

* local continual learning metrics
* federated metrics
* F1 improvements
* average scores

Run:

```bash
python3 results_report.py
```

---

# Results

| Bank | Local F1 | Federated F1 |
| ---- | -------- | ------------ |
| A    | 0.6485   | 0.7363       |
| B    | 0.8800   | 0.9230       |
| C    | 0.9400   | 0.9521       |

---

# 🛠 Future Improvements

* FedProx
* Differential Privacy
* SHAP Explainability
* Graph Neural Networks
* Real-time Streaming with Kafka
* Dashboard UI

---
