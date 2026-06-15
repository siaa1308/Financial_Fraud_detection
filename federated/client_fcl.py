import flwr as fl
import joblib
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
# Replay Buffer
# ==================================================
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
        fraud_batch = random.sample(self.fraud, min(len(self.fraud), half))
        normal_batch = random.sample(self.normal, min(len(self.normal), half))
        batch = fraud_batch + normal_batch
        random.shuffle(batch)

        if len(batch) == 0:
            return None, None

        X = np.array([b[0] for b in batch])
        y = np.array([b[1] for b in batch])
        return X, y


# ==================================================
# Model
# ==================================================
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


# ==================================================
# Flower Utils
# ==================================================
def get_parameters(model):
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state, strict=True)


# ==================================================
# Task Split
# ==================================================
def make_tasks(X, y, n_tasks=3):
    idx = np.arange(len(y))
    splits = np.array_split(idx, n_tasks)
    return [(X[s], y[s]) for s in splits]


# ==================================================
# Train Function
# ==================================================
def train_local(model, X, y, epochs=2):
    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)

    pos_weight = torch.tensor([(len(y)-y.sum().item())/y.sum().item()]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()


# ==================================================
# Evaluate
# ==================================================
def evaluate(model, X, y, threshold=0.3):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(model(X)).cpu().numpy().flatten()

    pred = (probs >= threshold).astype(int)

    return {
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
    }


# ==================================================
# Flower Client
# ==================================================
class FCLClient(fl.client.NumPyClient):
    def __init__(self, path):
        data = joblib.load(path)

        self.X_train = data["X_train"].toarray() if hasattr(data["X_train"], "toarray") else data["X_train"]
        self.y_train = data["y_train"]

        self.X_test = data["X_test"].toarray() if hasattr(data["X_test"], "toarray") else data["X_test"]
        self.y_test = data["y_test"]

        self.tasks = make_tasks(self.X_train, self.y_train, n_tasks=3)
        self.buffer = ReplayBuffer(2000)

        self.model = MLP(self.X_train.shape[1]).to(DEVICE)

        self.current_task = 0

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        # Train only next task each round
        X_new, y_new = self.tasks[self.current_task]

        X_old, y_old = self.buffer.sample(256)

        if X_old is not None:
            X_mix = np.vstack([X_new, X_old])
            y_mix = np.hstack([y_new, y_old])
        else:
            X_mix, y_mix = X_new, y_new

        train_local(self.model, X_mix, y_mix, epochs=2)

        self.buffer.add(X_new, y_new)

        if self.current_task < len(self.tasks) - 1:
            self.current_task += 1

        return get_parameters(self.model), len(X_mix), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)

        metrics = evaluate(self.model, self.X_test, self.y_test)

        print("\nClient Metrics:", metrics)

        return 0.0, len(self.X_test), metrics


# ==================================================
# Run
# ==================================================
if __name__ == "__main__":
    import sys

    path = sys.argv[1]

    client = FCLClient(path)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client
    )