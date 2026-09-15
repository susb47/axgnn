import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import os

# -------------------
# CONFIG
# -------------------
INPUT_FILE = r"D:\Research\Topics\www\Datasets\TCGA Liver Cancer (LIHC) Clinical Dataset\TCGA-LIHC_Clinical_Staging_Dataset__Standardized_Stages_.csv"
OUT_DIR = r"D:\Research\Topics\www\Datasets\TCGA Liver Cancer (LIHC) Clinical Dataset\graph"
OUT_FILE = os.path.join(OUT_DIR, "tcga_lihc_graph.pt")

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------
# Load dataset
# -------------------
df = pd.read_csv(INPUT_FILE)

# Drop useless index col if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Handle Age
df["Age_at_Diagnosis"] = pd.to_numeric(df["Age_at_Diagnosis"], errors="coerce")
df["Age_at_Diagnosis"] = df["Age_at_Diagnosis"].fillna(df["Age_at_Diagnosis"].mean())

# Encode categorical
encoders = {}
for col in ["Gender", "Race", "Vital_Status", "Tumor_Stage"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# -------------------
# Build graphs (1 per patient)
# -------------------
graphs = []

for i, row in df.iterrows():
    # Node features = each attribute as its own node
    # Nodes: [Age, Gender, Race, Vital_Status]
    x = torch.tensor([
        [row["Age_at_Diagnosis"], 0, 0, 0],
        [0, row["Gender"], 0, 0],
        [0, 0, row["Race"], 0],
        [0, 0, 0, row["Vital_Status"]],
    ], dtype=torch.float)

    # Node types (0=Age, 1=Gender, 2=Race, 3=Vital_Status)
    node_type = torch.tensor([0,1,2,3], dtype=torch.long)

    # Fully connect the nodes (undirected edges)
    edge_index = []
    edge_type = []
    for s in range(4):
        for t in range(s+1,4):
            edge_index.append([s,t])
            edge_index.append([t,s])
            edge_type.append(0)  # single edge type "attribute link"
            edge_type.append(0)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    # Label = Tumor Stage
    y = torch.tensor([row["Tumor_Stage"]], dtype=torch.long)

    g = Data(x=x, edge_index=edge_index, edge_type=edge_type, node_type=node_type, y=y)
    graphs.append(g)

# -------------------
# Save as list of graphs
# -------------------
torch.save(graphs, OUT_FILE)

print(f"✅ Saved {len(graphs)} patient graphs to {OUT_FILE}")
print("Example graph:", graphs[0])
