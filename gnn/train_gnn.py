import joblib
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import classification_report, f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


# Load graph
data = joblib.load("gnn/graphs/bankA_graph.pkl")
data = data.to(DEVICE)

# Train/test masks
n = data.num_nodes
perm = torch.randperm(n)

train_size = int(0.8 * n)

train_mask = torch.zeros(n, dtype=torch.bool)
test_mask = torch.zeros(n, dtype=torch.bool)

train_mask[perm[:train_size]] = True
test_mask[perm[train_size:]] = True

data.train_mask = train_mask.to(DEVICE)
data.test_mask = test_mask.to(DEVICE)

# Model
model = GraphSAGE(data.num_features).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Class weights
num_pos = int(data.y.sum())
num_neg = len(data.y) - num_pos
weights = torch.tensor([1.0, num_neg / max(num_pos,1)]).to(DEVICE)

criterion = torch.nn.CrossEntropyLoss(weight=weights)

# Train
for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss {loss.item():.4f}")

# Evaluate
model.eval()
out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)

y_true = data.y[data.test_mask].cpu().numpy()
y_pred = pred[data.test_mask].cpu().numpy()

print("\nF1:", f1_score(y_true, y_pred))
print(classification_report(y_true, y_pred))