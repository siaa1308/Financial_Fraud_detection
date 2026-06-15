import flwr as fl
import joblib
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score, precision_score, recall_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================
# Model
# =========================================
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 64)
        self.conv2 = SAGEConv(64, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# =========================================
# Replay Buffer
# =========================================
class ReplayBuffer:
    def __init__(self, max_size=500):
        self.memory = []
        self.max_size = max_size

    def add(self, idx_list):
        for i in idx_list:
            self.memory.append(int(i))
            if len(self.memory) > self.max_size:
                self.memory.pop(0)

    def sample(self, n=100):
        if len(self.memory) == 0:
            return []
        return random.sample(self.memory, min(n, len(self.memory)))


# =========================================
# Params
# =========================================
def get_parameters(model):
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def set_parameters(model, parameters):
    keys = model.state_dict().keys()
    state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state, strict=True)


# =========================================
# Client
# =========================================
class GNNClient(fl.client.NumPyClient):
    def __init__(self, graph_path):
        self.data = joblib.load(graph_path).to(DEVICE)

        self.model = GraphSAGE(self.data.num_features).to(DEVICE)

        # Node split into tasks
        n = self.data.num_nodes
        perm = torch.randperm(n).cpu().numpy()

        self.tasks = np.array_split(perm[:int(0.8*n)], 3)
        self.test_idx = perm[int(0.8*n):]

        self.task_id = 0
        self.buffer = ReplayBuffer()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        current_nodes = self.tasks[self.task_id]
        replay_nodes = self.buffer.sample(100)

        train_nodes = np.unique(
            np.concatenate([current_nodes, replay_nodes])
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        y = self.data.y[train_nodes]
        num_pos = int(y.sum().item())
        num_neg = len(y) - num_pos
        weights = torch.tensor(
            [1.0, num_neg / max(num_pos, 1)],
            device=DEVICE
        )

        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        self.model.train()

        for _ in range(20):
            optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = criterion(out[train_nodes], self.data.y[train_nodes])
            loss.backward()
            optimizer.step()

        self.buffer.add(current_nodes)

        if self.task_id < 2:
            self.task_id += 1

        return get_parameters(self.model), len(train_nodes), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)

        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out.argmax(dim=1)

        y_true = self.data.y[self.test_idx].cpu().numpy()
        y_pred = pred[self.test_idx].cpu().numpy()

        metrics = {
            "f1": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
        }

        print("Metrics:", metrics)

        return 0.0, len(self.test_idx), metrics


# =========================================
# Run
# =========================================
if __name__ == "__main__":
    import sys

    path = sys.argv[1]

    client = GNNClient(path)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=client
    )