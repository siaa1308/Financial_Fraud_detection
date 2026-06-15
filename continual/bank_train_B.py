import joblib
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, max_size=2000):
        self.max_size = max_size
        self.fraud = []
        self.normal = []

    def add(self, X, y):
        for xi, yi in zip(X, y):
            if yi == 1:
                self.fraud.append((xi, yi))
                if len(self.fraud) > self.max_size // 2:
                    self.fraud.pop(0)
            else:
                self.normal.append((xi, yi))
                if len(self.normal) > self.max_size // 2:
                    self.normal.pop(0)

    def sample(self, n=256):
        half = n // 2
        f = random.sample(self.fraud, min(len(self.fraud), half))
        nrm = random.sample(self.normal, min(len(self.normal), half))
        batch = f + nrm
        random.shuffle(batch)

        if len(batch) == 0:
            return None, None

        X = np.array([b[0] for b in batch])
        y = np.array([b[1] for b in batch])
        return X, y


# -------------------------
# Model
# -------------------------
class MLP(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Create Time Tasks
# -------------------------
def make_tasks(X, y, n_tasks=3):
    idx = np.arange(len(y))
    splits = np.array_split(idx, n_tasks)
    tasks = []

    for s in splits:
        tasks.append((X[s], y[s]))
    return tasks


# -------------------------
# Train One Task
# -------------------------
def train_task(model, X, y, criterion, optimizer, epochs=5):
    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)

    model.train()
    for ep in range(epochs):
        total = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()

        print(f"Epoch {ep+1}: Loss={total:.4f}")


# -------------------------
# Evaluate
# -------------------------
def evaluate(model, X_test, y_test, threshold=0.4):
    model.eval()

    X = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(model(X)).cpu().numpy().flatten()

    preds = (probs >= threshold).astype(int)

    print("\nPrecision:", precision_score(y_test, preds))
    print("Recall   :", recall_score(y_test, preds))
    print("F1       :", f1_score(y_test, preds))
    print()
    print(classification_report(y_test, preds))


# -------------------------
# Main
# -------------------------
def run_bank(path):
    data = joblib.load(path)

    X_train = data["X_train"].toarray() if hasattr(data["X_train"], "toarray") else data["X_train"]
    y_train = data["y_train"]

    X_test = data["X_test"].toarray() if hasattr(data["X_test"], "toarray") else data["X_test"]
    y_test = data["y_test"]

    input_dim = X_train.shape[1]

    model = MLP(input_dim).to(DEVICE)

    pos_weight = torch.tensor([(len(y_train)-sum(y_train))/sum(y_train)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    tasks = make_tasks(X_train, y_train, n_tasks=3)
    buffer = ReplayBuffer(max_size=2000)

    for i, (X_new, y_new) in enumerate(tasks):
        print(f"\n=== Training Task {i+1} ===")

        X_old, y_old = buffer.sample(256)

        if X_old is not None:
            X_mix = np.vstack([X_new, X_old])
            y_mix = np.hstack([y_new, y_old])
        else:
            X_mix, y_mix = X_new, y_new

        train_task(model, X_mix, y_mix, criterion, optimizer, epochs=5)

        buffer.add(X_new, y_new)

    print("\n=== Final Evaluation ===")
    evaluate(model, X_test, y_test)

    return model


if __name__ == "__main__":
    run_bank("processed/bankB.pkl")