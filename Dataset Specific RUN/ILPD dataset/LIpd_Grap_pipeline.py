import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os, json

# --- Visualization imports ---
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import matplotlib.patches as mpatches

# Node and Edge type enums
NODE_TYPE = {"organ":0, "tissue":1, "symptom":2, "patient":3}
EDGE_TYPE = {"part_of":0, "manifests_in":1, "co_dep":2, "adjacent_to":3, "has_finding":4}

# Column names (confirmed from your CSV)
SYMPTOMS = ["TB","DB","Alkphos","SGPT","SGOT","TP","ALB","A/G","Age","Gender"]

# ---------- Graph Builder ----------
def build_graph(record):
    nodes = ["liver"] + SYMPTOMS + ["patient"]
    node_type = [NODE_TYPE["organ"]] + [NODE_TYPE["symptom"]]*len(SYMPTOMS) + [NODE_TYPE["patient"]]

    feats = []
    feats.append(np.array([0.0], dtype=np.float32))  # organ placeholder
    for s in SYMPTOMS:
        val = record[s]
        if s == "Gender":
            if str(val).strip().lower().startswith("m"):
                feats.append(np.array([1.0], dtype=np.float32))  # Male
            else:
                feats.append(np.array([0.0], dtype=np.float32))  # Female
        else:
            feats.append(np.array([float(val)], dtype=np.float32))
    feats.append(np.array([0.0], dtype=np.float32))  # patient placeholder

    x = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32)

    E, T = [], []
    for i in range(1, 1+len(SYMPTOMS)):
        E += [[0,i],[i,0]]
        T += [EDGE_TYPE["manifests_in"], EDGE_TYPE["manifests_in"]]
    p = len(nodes)-1
    for i in range(1, 1+len(SYMPTOMS)):
        E += [[p,i],[i,p]]
        T += [EDGE_TYPE["has_finding"], EDGE_TYPE["has_finding"]]

    edge_index = torch.tensor(E, dtype=torch.long).t().contiguous()
    edge_type  = torch.tensor(T, dtype=torch.long)
    y = torch.tensor([1 if int(record["Dataset"])==1 else 0], dtype=torch.long)

    return Data(
        x=x, edge_index=edge_index, edge_type=edge_type,
        node_type=torch.tensor(node_type, dtype=torch.long), y=y,
        node_labels=nodes
    )

# ---------- Beautiful PNG Visualization ----------
def visualize_graph_beautiful_static(data, save_path="graph_beauty.png", title="ILPD Graph"):
    G = nx.Graph()
    labels, edge_labels = {}, {}
    for i, name in enumerate(data.node_labels):
        G.add_node(i, label=name, ntype=int(data.node_type[i]))
        labels[i] = name

    for idx, (u,v) in enumerate(data.edge_index.t().tolist()):
        G.add_edge(u,v)
        etype = int(data.edge_type[idx])
        etype_name = [k for k,vv in EDGE_TYPE.items() if vv == etype][0]
        edge_labels[(u,v)] = etype_name

    color_map = []
    for n in G.nodes(data=True):
        ntype = n[1]['ntype']
        if ntype == NODE_TYPE["organ"]:   color_map.append("#E74C3C")
        elif ntype == NODE_TYPE["symptom"]: color_map.append("#3498DB")
        elif ntype == NODE_TYPE["patient"]: color_map.append("#F1C40F")
        else: color_map.append("#95A5A6")

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(9,7))
    nx.draw(G, pos, with_labels=True, labels=labels,
            node_color=color_map, node_size=1200,
            font_size=9, font_weight="bold", edge_color="gray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    patches = [
        mpatches.Patch(color="#E74C3C", label="Organ"),
        mpatches.Patch(color="#3498DB", label="Symptom/Lab"),
        mpatches.Patch(color="#F1C40F", label="Patient"),
    ]
    plt.legend(handles=patches, loc="upper right")
    plt.title(title, fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] PNG visualization → {save_path}")

# ---------- Interactive HTML Visualization ----------
def visualize_graph_interactive_pyvis(data, html_path="graph_interactive.html"):
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="#333")
    net.toggle_physics(True)
    xvals = data.x.squeeze(-1).cpu().numpy()

    for i, name in enumerate(data.node_labels):
        ntype = int(data.node_type[i])
        if ntype == NODE_TYPE["organ"]: color = "#E74C3C"
        elif ntype == NODE_TYPE["symptom"]: color = "#3498DB"
        elif ntype == NODE_TYPE["patient"]: color = "#F1C40F"
        else: color = "#95A5A6"
        val = float(xvals[i])
        net.add_node(i, label=name, title=f"{name}<br>Value: {val:.2f}", color=color, size=20)

    eidx = data.edge_index.t().cpu().numpy()
    etypes = data.edge_type.cpu().numpy()
    for k,(u,v) in enumerate(eidx):
        et = int(etypes[k])
        label = [k for k,vv in EDGE_TYPE.items() if vv == et][0]
        net.add_edge(int(u),int(v), title=label, color="#7f8c8d")

    net.write_html(html_path)
    print(f"[SAVED] Interactive HTML → {html_path}")

# ---------- Main ----------
def main():
    df = pd.read_csv(r"D:\Research\Topics\www\Datasets\ilpd+indian+liver+patient+dataset\Indian_Liver_Patient_Dataset_Updated.csv", header=None)
    df.columns = ["Age","Gender","TB","DB","Alkphos","SGPT","SGOT","TP","ALB","A/G","Dataset"]

    out_dir = r"D:\Research\Topics\www\Datasets\ilpd+indian+liver+patient+dataset\Output\latest output"
    os.makedirs(out_dir, exist_ok=True)

    all_graphs = []
    for i,row in df.iterrows():
        rec = row.to_dict()
        g = build_graph(rec)
        all_graphs.append(g)

        # Visualization for first case
        if i == 0:
            visualize_graph_beautiful_static(g, save_path=os.path.join(out_dir,"ilpd_00000.png"))
            visualize_graph_interactive_pyvis(g, html_path=os.path.join(out_dir,"ilpd_00000.html"))

    # Save all graphs in one .pt file
    pt_path = os.path.join(out_dir, "ilpd_dataset.pt")
    torch.save(all_graphs, pt_path)

    print(f"[DONE] Saved {len(all_graphs)} ILPD graphs in → {pt_path}")


if __name__ == "__main__":
    main()
