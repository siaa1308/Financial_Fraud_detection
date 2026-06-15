import pandas as pd

# ======================================
# Results Data
# ======================================

df_local = pd.DataFrame([
    ["Bank A", 0.4817, 0.9920, 0.6485],
    ["Bank B", 0.7900, 0.9900, 0.8800],
    ["Bank C", 0.8900, 1.0000, 0.9400],
], columns=["Bank", "Precision", "Recall", "F1"])

df_tab = pd.DataFrame([
    ["Bank A", 0.5843, 0.9950, 0.7363],
    ["Bank B", 0.8630, 0.9919, 0.9230],
    ["Bank C", 0.9153, 0.9919, 0.9521],
], columns=["Bank", "Precision", "Recall", "F1"])

df_gnn = pd.DataFrame([
    ["Bank A", 0.9589, 0.8929, 0.9248],
    ["Bank B", 0.9540, 0.9110, 0.9320],
    ["Bank C", 0.9293, 0.9091, 0.9191],
], columns=["Bank", "Precision", "Recall", "F1"])

# NEW XGBOOST RESULTS
df_xgb = pd.DataFrame([
    ["Bank A", 0.6000, 1.0000, 0.7500],
    ["Bank B", 0.7800, 1.0000, 0.8800],
    ["Bank C", 0.8200, 1.0000, 0.9000],
], columns=["Bank", "Precision", "Recall", "F1"])


# ======================================
# Show Individual Results
# ======================================

print("\n" + "="*90)
print("LOCAL CONTINUAL LEARNING RESULTS")
print("="*90)
print(df_local.to_string(index=False))

print("\n" + "="*90)
print("TABULAR FEDERATED CONTINUAL LEARNING RESULTS")
print("="*90)
print(df_tab.to_string(index=False))

print("\n" + "="*90)
print("GRAPH FEDERATED CONTINUAL LEARNING RESULTS")
print("="*90)
print(df_gnn.to_string(index=False))

print("\n" + "="*90)
print("XGBOOST BASELINE RESULTS")
print("="*90)
print(df_xgb.to_string(index=False))


# ======================================
# Full Comparison Table
# ======================================

rows = []

for i in range(len(df_local)):
    rows.append({
        "Bank": df_local.loc[i, "Bank"],

        "Local_P": df_local.loc[i, "Precision"],
        "Local_R": df_local.loc[i, "Recall"],
        "Local_F1": df_local.loc[i, "F1"],

        "Tab_P": df_tab.loc[i, "Precision"],
        "Tab_R": df_tab.loc[i, "Recall"],
        "Tab_F1": df_tab.loc[i, "F1"],

        "GNN_P": df_gnn.loc[i, "Precision"],
        "GNN_R": df_gnn.loc[i, "Recall"],
        "GNN_F1": df_gnn.loc[i, "F1"],

        "XGB_P": df_xgb.loc[i, "Precision"],
        "XGB_R": df_xgb.loc[i, "Recall"],
        "XGB_F1": df_xgb.loc[i, "F1"],
    })

df_compare = pd.DataFrame(rows)

print("\n" + "="*150)
print("FULL MODEL COMPARISON")
print("="*150)
print(df_compare.to_string(index=False))


# ======================================
# Average Metrics
# ======================================

summary = pd.DataFrame([
    ["Local CL", df_local["Precision"].mean(), df_local["Recall"].mean(), df_local["F1"].mean()],
    ["Tabular FCL", df_tab["Precision"].mean(), df_tab["Recall"].mean(), df_tab["F1"].mean()],
    ["Graph FCL", df_gnn["Precision"].mean(), df_gnn["Recall"].mean(), df_gnn["F1"].mean()],
    ["XGBoost", df_xgb["Precision"].mean(), df_xgb["Recall"].mean(), df_xgb["F1"].mean()],
], columns=["Model", "Avg Precision", "Avg Recall", "Avg F1"])

print("\n" + "="*90)
print("AVERAGE METRICS ACROSS BANKS")
print("="*90)
print(summary.round(4).to_string(index=False))