import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import joblib
import os

os.makedirs("gnn/graphs", exist_ok=True)


def build_graph(csv_path, save_path):
    df = pd.read_csv(csv_path)

    # Keep useful columns
    df = df[[
        "orig_acct", "bene_acct",
        "base_amt", "is_sar"
    ]].copy()

    df["base_amt"] = df["base_amt"].fillna(df["base_amt"].median())

    # Unique accounts -> node ids
    accounts = pd.Index(
        pd.concat([df["orig_acct"], df["bene_acct"]]).unique()
    )

    acc2id = {acc: i for i, acc in enumerate(accounts)}

    num_nodes = len(accounts)

    # ----------------------------------
    # Build Node Features
    # ----------------------------------
    out_count = df.groupby("orig_acct").size()
    in_count = df.groupby("bene_acct").size()

    out_amt = df.groupby("orig_acct")["base_amt"].sum()
    in_amt = df.groupby("bene_acct")["base_amt"].sum()

    X = np.zeros((num_nodes, 4), dtype=np.float32)

    for acc, idx in acc2id.items():
        X[idx, 0] = out_count.get(acc, 0)
        X[idx, 1] = in_count.get(acc, 0)
        X[idx, 2] = out_amt.get(acc, 0)
        X[idx, 3] = in_amt.get(acc, 0)

    # log scale amounts
    X[:, 2:] = np.log1p(X[:, 2:])

    # ----------------------------------
    # Node Labels
    # If account involved in fraud tx => 1
    # ----------------------------------
    y = np.zeros(num_nodes, dtype=np.int64)

    fraud_df = df[df["is_sar"] == 1]

    fraud_accounts = set(fraud_df["orig_acct"]).union(
        set(fraud_df["bene_acct"])
    )

    for acc in fraud_accounts:
        y[acc2id[acc]] = 1

    # ----------------------------------
    # Edges
    # ----------------------------------
    src = df["orig_acct"].map(acc2id).values
    dst = df["bene_acct"].map(acc2id).values

    edge_index = torch.tensor(
        np.vstack([src, dst]),
        dtype=torch.long
    )

    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long)
    )

    joblib.dump(data, save_path)
    print(f"Saved graph -> {save_path}")
    print(data)


build_graph("data/Bank_A.csv", "gnn/graphs/bankA_graph.pkl")
build_graph("data/Bank_B.csv", "gnn/graphs/bankB_graph.pkl")
build_graph("data/Bank_C.csv", "gnn/graphs/bankC_graph.pkl")