import matplotlib.pyplot as plt
import numpy as np

banks = ["Bank A", "Bank B", "Bank C"]

local_f1 = [0.649, 0.880, 0.940]
tabular_fcl = [0.736, 0.923, 0.952]
graph_fcl = [0.925, 0.932, 0.919]

x = np.arange(len(banks))
w = 0.25

# -----------------------------------
# Chart 1: F1 Comparison
# -----------------------------------
plt.figure(figsize=(10,6))
plt.bar(x - w, local_f1, width=w, label="Local CL")
plt.bar(x, tabular_fcl, width=w, label="Tabular FCL")
plt.bar(x + w, graph_fcl, width=w, label="Graph FCL")

plt.xticks(x, banks)
plt.ylabel("F1 Score")
plt.ylim(0.5, 1.0)
plt.title("Model Comparison Across Banks")
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------------
# Chart 2: Graph FCL Metrics
# -----------------------------------
precision = [0.959, 0.954, 0.929]
recall = [0.893, 0.911, 0.909]
f1 = [0.925, 0.932, 0.919]

plt.figure(figsize=(10,6))
plt.plot(banks, precision, marker="o", label="Precision")
plt.plot(banks, recall, marker="s", label="Recall")
plt.plot(banks, f1, marker="^", label="F1")

plt.ylim(0.85, 1.0)
plt.title("Graph FCL Performance Metrics")
plt.ylabel("Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()