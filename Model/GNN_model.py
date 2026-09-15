import os, json, random, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             average_precision_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import pandas as pd
import seaborn as sns
from copy import deepcopy

# --------------------------
# CONFIG
# --------------------------
CONFIG = {
    "DATA_PATH": r"D:\Research\Topics\www\Datasets\TCGA Liver Cancer (LIHC) Clinical Dataset\graph\tcga_lihc_graph.pt",
    "OUT_DIR":  r"D:\Research\Topics\www\Datasets\TCGA Liver Cancer (LIHC) Clinical Dataset\Model Output",
    "EPOCHS": 200,
    "BATCH_SIZE": 32,
    "HIDDEN_DIM": 128,
    "NODE_TYPE_EMB": 8,
    "EDGE_TYPE_EMB": 8,
    "VAL_RATIO": 0.20,
    "EARLY_STOP": 2001,
    "CLASS_WEIGHT": True,
    "SEED": 42,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EXPLAIN_ON_VAL_K": 5
}

NODE_TYPE_LABELS = {0:"Organ", 1:"Tissue", 2:"Symptom", 3:"Patient"}

# --------------------------
# Helpers
# --------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def init_csv_logger(path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path,"w",newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f,writer

def log_metrics(writer, epoch, train_loss, train_acc, val_metrics):
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_metrics.get("loss", None),
        "val_acc": val_metrics["acc"],
        "val_auroc": val_metrics.get("auroc", None),
        "val_auprc": val_metrics.get("auprc", None)
    }
    writer.writerow(row)

def plot_training_curves(csv_path, out_dir):
    df = pd.read_csv(csv_path)

    # ----------------- Loss -----------------
    plt.figure(figsize=(8,5))
    if "train_loss" in df:
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss", color="red")
    if "val_loss" in df:
        plt.plot(df["epoch"], df["val_loss"], label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # ----------------- Accuracy -----------------
    plt.figure(figsize=(8,5))
    if "train_acc" in df:
        plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy", color="blue")
    if "val_acc" in df:
        plt.plot(df["epoch"], df["val_acc"], label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=150)
    plt.close()

    # ----------------- AUROC -----------------
    if "val_auroc" in df and df["val_auroc"].notna().any():
        plt.figure(figsize=(8,5))
        plt.plot(df["epoch"], df["val_auroc"], label="Val AUROC", color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("AUROC")
        plt.title("Validation AUROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "auroc_curve.png"), dpi=150)
        plt.close()

    # ----------------- AUPRC -----------------
    if "val_auprc" in df and df["val_auprc"].notna().any():
        plt.figure(figsize=(8,5))
        plt.plot(df["epoch"], df["val_auprc"], label="Val AUPRC", color="brown")
        plt.xlabel("Epoch")
        plt.ylabel("AUPRC")
        plt.title("Validation AUPRC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "auprc_curve.png"), dpi=150)
        plt.close()
    
    if ("val_auroc" in df and df["val_auroc"].notna().any()) or \
        ("val_auprc" in df and df["val_auprc"].notna().any()):
        
        plt.figure(figsize=(8,5))
        
        if "val_auroc" in df and df["val_auroc"].notna().any():
            plt.plot(df["epoch"], df["val_auroc"], label="Val AUROC", color="purple")
        
        if "val_auprc" in df and df["val_auprc"].notna().any():
            plt.plot(df["epoch"], df["val_auprc"], label="Val AUPRC", color="brown")
        
        plt.xlabel("Epoch")
        plt.ylabel("Score (0–1)")
        plt.title("Validation AUROC & AUPRC Curves")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "auroc_auprc_curve.png"), dpi=150)
        plt.close()


def plot_confusion_matrix(cm, class_names, title, save_path=None, normalize=False):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)
        fmt = ".2f"
    else:
        fmt = "d"
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.title(title + (" (normalized)" if normalize else ""))
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def agg_plot_bar(values_dict, title, ylabel, save_path):
    labels=list(values_dict.keys()); vals=[values_dict[k] for k in labels]
    plt.figure(figsize=(7,4))
    plt.bar(labels, vals)
    plt.title(title); plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def agg_write_csv(rows, save_path, fieldnames=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with open(save_path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)

def agg_plot_heatmap(edge_by_ntype, save_path, normalize_rows=False):
    types = sorted(set([t for k in edge_by_ntype.keys() for t in k])) if edge_by_ntype else []
    if not types:
        mat = np.zeros((4,4))
        types = [0,1,2,3]
    else:
        mat = np.zeros((max(types)+1, max(types)+1))
    for (s,d),vals in edge_by_ntype.items():
        mat[s,d]=np.mean(vals)
    if normalize_rows:
        mat = mat / (mat.sum(1, keepdims=True) + 1e-9)
    labels=[NODE_TYPE_LABELS.get(t,f"type{t}") for t in range(mat.shape[0])]
    plt.figure(figsize=(6,5))
    sns.heatmap(mat,xticklabels=labels,yticklabels=labels,annot=True,cmap="Blues",fmt=".2f")
    plt.title("Mean Edge Attention by Node-Type Pair" + (" (row-normalized)" if normalize_rows else ""))
    plt.xlabel("Dst Node Type"); plt.ylabel("Src Node Type")
    plt.tight_layout(); plt.savefig(save_path,dpi=150); plt.close()

def plot_deletion_curve(curve_dict, save_path, title="Faithfulness Curve"):
    probs = curve_dict["deletion_probs"]
    xs = list(range(len(probs)))
    plt.figure()
    plt.plot(xs, probs, marker="o")
    plt.xlabel("Deletion step")
    plt.ylabel("Pred prob (top class)")
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# --------------------------
# Dataset (single .pt)
# --------------------------
class PTGraphDataset(Dataset):
    def __init__(self, pt_path):
        super().__init__()
        self.graphs = torch.load(pt_path)  # list of Data objects

    def len(self): return len(self.graphs)

    def get(self, idx):
        d = self.graphs[idx]
        lm = torch.zeros(d.x.size(0))
        lm[(d.node_type==0)|(d.node_type==1)] = 1.0
        d.landmark_mask = lm
        return d

# --------------------------
# Model
# --------------------------
class EdgeAwareConv(MessagePassing):
    def __init__(self,in_dim,out_dim,edge_type_vocab,edge_emb_dim):
        super().__init__(aggr="mean")
        self.edge_emb=nn.Embedding(edge_type_vocab,edge_emb_dim)
        self.msg_mlp=nn.Sequential(nn.Linear(in_dim+edge_emb_dim,out_dim), nn.ReLU(), nn.Linear(out_dim,out_dim))
        self.att_fc=nn.Linear(out_dim,1)
        self._last_alpha=None
    def forward(self,x,edge_index,edge_type,landmark_gate=None):
        return self.propagate(edge_index,x=x,edge_type=edge_type,landmark_gate=landmark_gate)
    def message(self,x_j,edge_type,index,landmark_gate):
        et=self.edge_emb(edge_type)
        m=self.msg_mlp(torch.cat([x_j,et],-1))
        a=self.att_fc(m).squeeze(-1)
        a=pyg_softmax(a,index)
        if landmark_gate is not None: a=a*landmark_gate
        self._last_alpha=a
        return m*a.unsqueeze(-1)

class LandmarkGate(nn.Module):
    def __init__(self,nt_dim=8,hid=32):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(2*nt_dim+2,hid), nn.ReLU(), nn.Linear(hid,1), nn.Sigmoid())
    def forward(self,nt_emb,edge_index,lm_mask):
        s,d=edge_index
        g=torch.stack([lm_mask[s],lm_mask[d]],-1)
        return self.mlp(torch.cat([nt_emb[s],nt_emb[d],g],-1)).squeeze(-1)

class AXGNN_XAI(nn.Module):
    def __init__(self,in_dim,hid,num_classes,node_vocab,nt_emb,edge_vocab,et_emb):
        super().__init__()
        self.nt_emb=nn.Embedding(node_vocab,nt_emb)
        self.in_proj=nn.Linear(in_dim+nt_emb,hid)
        self.gate=LandmarkGate(nt_dim=nt_emb)
        self.conv1=EdgeAwareConv(hid,hid,edge_vocab,et_emb)
        self.conv2=EdgeAwareConv(hid,hid,edge_vocab,et_emb)
        self.fc=nn.Linear(hid,num_classes)
        self.last_alpha=None; self.last_gate=None
    def forward(self,d):
        nt=self.nt_emb(d.node_type)
        h=self.in_proj(torch.cat([d.x,nt],1)).relu()
        g=self.gate(nt,d.edge_index,d.landmark_mask)
        h=self.conv1(h,d.edge_index,d.edge_type,g).relu()
        h=self.conv2(h,d.edge_index,d.edge_type,g).relu()
        self.last_alpha=self.conv2._last_alpha; self.last_gate=g
        batch=getattr(d,"batch",torch.zeros(h.size(0),dtype=torch.long,device=h.device))
        g=global_mean_pool(h,batch)
        return self.fc(g)

# --------------------------
# XAI utilities
# --------------------------
def get_edge_attention_and_gate(m, d):
    with torch.no_grad():
        _ = m(d)
        alpha = m.last_alpha.detach().cpu().numpy() if m.last_alpha is not None else None
        gate  = m.last_gate.detach().cpu().numpy()  if m.last_gate  is not None else None
    return alpha, gate

def grad_x_input_saliency(m, d):
    d = d.clone()
    d.x = d.x.clone().requires_grad_(True)
    logits = m(d)
    yhat = logits.softmax(1)[:, logits.argmax(1)]
    yhat.sum().backward()
    return (d.x.grad * d.x).abs().sum(1).detach().cpu().numpy()

def counterfactual_search_minimal_feature_delta(
    model, data, max_steps=200, step=0.1, topk_nodes=5, target_class=None
):
    model.eval()
    sal = grad_x_input_saliency(model, data)
    idx = np.argsort(-sal)[:topk_nodes]

    x0 = data.x.detach().clone()
    x  = x0.clone().requires_grad_(True)
    opt = torch.optim.SGD([x], lr=step)

    with torch.no_grad():
        orig = int(model(data).argmax(1).item())
    tgt = int(1 - orig) if target_class is None else int(target_class)

    for _ in range(max_steps):
        opt.zero_grad()
        data.x = x
        logits = model(data)
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([tgt], device=x.device))
        loss.backward()
        with torch.no_grad():
            mask = torch.zeros_like(x); mask[idx] = 1.0
            x.data = x.data - step * x.grad.data * mask
            x.data = torch.clip(x.data, x0.min() - 3.0, x0.max() + 3.0)
        if logits.argmax(1).item() == tgt:
            break

    delta = (x.detach() - x0).abs().sum().item()
    data.x = x0
    return {"orig_class": orig, "target_class": tgt, "l1_change": float(delta), "changed_nodes": idx.tolist()}

def faithfulness_deletion_curve(model, data, mode="node", steps=10):
    base_logits = model(data)
    base_cls = int(base_logits.argmax(1).item())
    base_prob = float(base_logits.softmax(1)[0, base_cls].item())
    probs = [base_prob]

    if mode == "node":
        sal = grad_x_input_saliency(model, data)
        order = np.argsort(-sal)
        d = deepcopy(data)
        for t in np.array_split(order, steps):
            d.x[t] = 0.0
            p = float(model(d).softmax(1)[0, base_cls].item())
            probs.append(p)
    else:
        _ = model(data)
        alpha = model.last_alpha.detach().cpu().numpy()
        order = np.argsort(-alpha)
        E = data.edge_index.size(1)
        for t in np.array_split(order, steps):
            keep = torch.ones(E, dtype=torch.bool, device=data.edge_index.device)
            keep[t] = False
            d_new = deepcopy(data)
            d_new.edge_index = data.edge_index[:, keep].clone().long()
            d_new.edge_type  = data.edge_type[keep].clone().long()
            if d_new.edge_index.numel() > 0:
                max_idx = int(d_new.edge_index.max().item())
                assert max_idx < d_new.x.size(0), f"Invalid edge_index: {max_idx} >= {d_new.x.size(0)}"
            p = float(model(d_new).softmax(1)[0, base_cls].item())
            probs.append(p)
    return {"base_prob": base_prob, "deletion_probs": probs}

def visualize_graph_explanation(
    d, node_scores, edge_scores, cf_nodes=None, faith_node_scores=None,
    save_png=None, save_html=None, title="Graph"
):
    N, E = d.x.size(0), d.edge_index.size(1)
    node_scores = (node_scores - np.min(node_scores)) / (np.max(node_scores) + 1e-9)
    if edge_scores is None: edge_scores = np.ones(E)
    edge_scores = (edge_scores - np.min(edge_scores)) / (np.max(edge_scores) + 1e-9)
    if faith_node_scores is None: faith_node_scores = np.ones(N)
    else:
        faith_node_scores = (faith_node_scores - np.min(faith_node_scores)) / (np.max(faith_node_scores) + 1e-9)

    G = nx.Graph()
    for j in range(E):
        s = int(d.edge_index[0,j]); t = int(d.edge_index[1,j])
        G.add_edge(s, t, weight=edge_scores[j])
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(7,7))

    for t,shape in [(0,"o"),(1,"s"),(2,"^"),(3,"D")]:
        idx = [i for i in range(N) if int(d.node_type[i]) == t]
        if idx:
            border = ["green" if cf_nodes is not None and i in cf_nodes else "black" for i in idx]
            sizes  = [300 + 400*faith_node_scores[i] for i in idx]
            nx.draw_networkx_nodes(
                G, pos, nodelist=idx,
                node_color=[node_scores[i] for i in idx],
                cmap=plt.cm.Reds, node_shape=shape,
                node_size=sizes, edgecolors=border, linewidths=2,
                label=NODE_TYPE_LABELS.get(t, f"type{t}")
            )
    nx.draw_networkx_edges(G, pos, width=[1+4*edge_scores[j] for j in range(E)], edge_color="blue", alpha=0.4)
    labels = {i:f"{NODE_TYPE_LABELS.get(int(d.node_type[i]),'?')}_{i}" for i in range(N)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.legend(loc="best"); plt.title(title); plt.axis("off")
    if save_png:
        plt.savefig(save_png, dpi=150, bbox_inches="tight")
        plt.close()
    if save_html:
        net = Network(height="600px", width="100%", notebook=False)
        for i in range(N):
            color = plt.cm.Reds(node_scores[i])
            col = f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.9)"
            net.add_node(i, label=labels[i], color=col, size=int(20+30*faith_node_scores[i]))
        for j in range(E):
            s, t = int(d.edge_index[0,j]), int(d.edge_index[1,j])
            net.add_edge(s, t, value=1+4*edge_scores[j])
        net.write_html(save_html)

# --------------------------
# Evaluation (shared)
# --------------------------
def evaluate(m, loader, device, num_classes, criterion=None, return_preds=False):
    m.eval()
    all_logits, all_y, losses = [], [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            logits = m(b)
            y = b.y.view(-1).long()
            if criterion is not None:
                losses.append(criterion(logits, y).item())
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())
    logits = torch.cat(all_logits)
    y = torch.cat(all_y).numpy()
    y_pred = logits.argmax(1).numpy()
    metrics = {"acc": accuracy_score(y, y_pred)}
    if num_classes == 2:
        probs = logits.softmax(1)[:,1].numpy()
        metrics["auroc"] = roc_auc_score(y, probs)
        metrics["auprc"] = average_precision_score(y, probs)
    if losses:
        metrics["loss"] = float(np.mean(losses))
    if return_preds:
        return metrics, y, y_pred
    return metrics

# --------------------------
# Main
# --------------------------
def main():
    cfg = CONFIG
    set_seed(cfg["SEED"])
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)

    dataset = PTGraphDataset(cfg["DATA_PATH"])
    labels = np.array([int(g.y.item()) for g in dataset.graphs])
    idxs = np.arange(len(dataset))

    # ---------------- Train / Validation Split ----------------
    tr, va = train_test_split(
        idxs,
        test_size=cfg["VAL_RATIO"],
        stratify=labels,
        random_state=cfg["SEED"]
    )

    tr_loader = DataLoader(torch.utils.data.Subset(dataset, tr),
                           batch_size=cfg["BATCH_SIZE"], shuffle=True)
    va_loader = DataLoader(torch.utils.data.Subset(dataset, va),
                           batch_size=cfg["BATCH_SIZE"])

    s = dataset.get(0)
    in_dim = s.x.size(1)
    num_classes = int(labels.max()) + 1

    model = AXGNN_XAI(
        in_dim,
        cfg["HIDDEN_DIM"],
        num_classes,
        int(s.node_type.max()) + 1,
        cfg["NODE_TYPE_EMB"],
        int(s.edge_type.max()) + 1,
        cfg["EDGE_TYPE_EMB"]
    ).to(cfg["DEVICE"])

    if cfg["CLASS_WEIGHT"] and num_classes == 2:
        counts = np.bincount(labels[tr])
        w = counts.sum() / (len(counts) * counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32, device=cfg["DEVICE"])
        )
    else:
        criterion = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    csv_path = os.path.join(cfg["OUT_DIR"], "metrics.csv")
    logf, writer = init_csv_logger(
        csv_path,
        ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auroc", "val_auprc"]
    )

    best = -1
    pat = 0
    best_val_acc = -1
    for ep in range(cfg["EPOCHS"]):
        print(f"[EPOCH {ep+1}]:")
        model.train()
        losses = []
        correct = 0
        total = 0
        for b in tr_loader:
            b = b.to(cfg["DEVICE"])
            opt.zero_grad()
            logits = model(b)
            target = b.y.view(-1).long()
            loss = criterion(logits, target)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pred = logits.argmax(1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        train_loss = float(np.mean(losses))
        train_acc = correct / total

        val = evaluate(model, va_loader, cfg["DEVICE"], num_classes, criterion)
        log_metrics(writer, ep, train_loss, train_acc, val)

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val.get('loss',0):.4f} | Val Acc: {val['acc']:.4f}"
            + (f" | Val AUROC: {val['auroc']:.4f} | Val AUPRC: {val['auprc']:.4f}"
               if num_classes == 2 else "")
        )

        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]

        metric = val.get("auroc", val["acc"])
        if metric > best:
            best = metric
            pat = 0
            torch.save(model.state_dict(), os.path.join(cfg["OUT_DIR"], "best.pt"))
        else:
            pat += 1
        if pat >= cfg["EARLY_STOP"]:
            break

    logf.close()
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    plot_training_curves(csv_path, cfg["OUT_DIR"])

    # ---------------- Validation Confusion Matrix ----------------
    model.load_state_dict(torch.load(os.path.join(cfg["OUT_DIR"], "best.pt"),
                                     map_location=cfg["DEVICE"]))
    val_metrics, y_val_true, y_val_pred = evaluate(
        model, va_loader, cfg["DEVICE"], num_classes, criterion, return_preds=True
    )
    cm_val = confusion_matrix(y_val_true, y_val_pred)
    print("Validation Confusion Matrix:\n", cm_val)
    plot_confusion_matrix(cm_val, class_names=[str(i) for i in range(num_classes)],
                          title="Validation Confusion Matrix",
                          save_path=os.path.join(cfg["OUT_DIR"], "val_confusion_matrix.png"))
    plot_confusion_matrix(cm_val, class_names=[str(i) for i in range(num_classes)],
                          title="Validation Confusion Matrix",
                          save_path=os.path.join(cfg["OUT_DIR"], "val_confusion_matrix_norm.png"),
                          normalize=True)

    # ---------------- Explanations on a few val graphs ----------------
    exp_dir = os.path.join(cfg["OUT_DIR"], "explanations")
    os.makedirs(exp_dir, exist_ok=True)
    agg_sal, agg_faith = {0: [], 1: [], 2: [], 3: []}, {0: [], 1: [], 2: [], 3: []}
    agg_attn_et, agg_edge_ntype = {}, {}

    picked = list(map(int, va[:cfg["EXPLAIN_ON_VAL_K"]]))  # first K from validation

    for i in picked:
        d = dataset.get(i).to(cfg["DEVICE"])
        with torch.no_grad():
            logits = model(d)
            pred = int(logits.argmax(1))
            prob = float(logits.softmax(1)[0, pred])

        alpha, gate = get_edge_attention_and_gate(model, d)
        sal = grad_x_input_saliency(model, d)
        cf = counterfactual_search_minimal_feature_delta(model, d)
        faith_nodes = faithfulness_deletion_curve(model, d, mode="node", steps=10)
        faith_edges = faithfulness_deletion_curve(model, d, mode="edge", steps=10)

        # Save deletion curves
        plot_deletion_curve(faith_nodes, os.path.join(exp_dir, f"faith_nodes_{i}.png"),
                            title=f"Node deletion {i}")
        plot_deletion_curve(faith_edges, os.path.join(exp_dir, f"faith_edges_{i}.png"),
                            title=f"Edge deletion {i}")

        # Simple faithfulness proxy
        base = faith_nodes["base_prob"]
        drop1 = max(0.0, base - faith_nodes["deletion_probs"][1]
                    if len(faith_nodes["deletion_probs"]) > 1 else 0.0)
        N = d.x.size(0)
        k = max(1, int(0.1 * N))
        top_idx = np.argsort(-sal)[:k]
        node_faith = np.zeros(N, dtype=np.float32)
        node_faith[top_idx] = drop1 / k

        # Aggregations
        nt = d.node_type.detach().cpu().numpy()
        for t in [0, 1, 2, 3]:
            idx_t = np.where(nt == t)[0]
            if idx_t.size > 0:
                agg_sal[t].append(float(np.mean(sal[idx_t])))
                agg_faith[t].append(float(np.mean(node_faith[idx_t])))
        if alpha is not None:
            et = d.edge_type.detach().cpu().numpy()
            for et_id in np.unique(et):
                m = float(np.mean(alpha[et == et_id]))
                agg_attn_et.setdefault(int(et_id), []).append(m)
            src_types = d.node_type[d.edge_index[0]].detach().cpu().numpy()
            dst_types = d.node_type[d.edge_index[1]].detach().cpu().numpy()
            for j in range(len(et)):
                key = (int(src_types[j]), int(dst_types[j]))
                agg_edge_ntype.setdefault(key, []).append(float(alpha[j]))

        # Save graph explanations
        visualize_graph_explanation(
            d.cpu(),
            node_scores=sal,
            edge_scores=alpha if alpha is not None else None,
            cf_nodes=cf["changed_nodes"],
            faith_node_scores=node_faith,
            save_png=os.path.join(exp_dir, f"graph_{i}.png"),
            save_html=os.path.join(exp_dir, f"graph_{i}.html"),
            title=f"Graph {i} pred={pred} p={prob:.2f}"
        )

        # JSON dump
        with open(os.path.join(exp_dir, f"explain_{i}.json"), "w") as f:
            json.dump({
                "graph_index": int(i),
                "pred_class": pred,
                "pred_prob": prob,
                "edge_attention": None if alpha is None else alpha.tolist(),
                "edge_gate": None if gate is None else gate.tolist(),
                "node_saliency": sal.tolist(),
                "counterfactual": cf,
                "faithfulness_nodes": faith_nodes,
                "faithfulness_edges": faith_edges
            }, f, indent=2)

    # -------- Aggregate XAI summaries --------
    sal_means = {NODE_TYPE_LABELS[t]: (np.mean(v) if v else 0.0) for t, v in agg_sal.items()}
    faith_means = {NODE_TYPE_LABELS[t]: (np.mean(v) if v else 0.0) for t, v in agg_faith.items()}
    attn_means = {f"edge_type_{k}": (np.mean(v) if v else 0.0) for k, v in agg_attn_et.items()}

    rows = []
    for t in [0, 1, 2, 3]:
        rows.append({"group": "node_type", "key": NODE_TYPE_LABELS[t],
                     "mean_saliency": sal_means[NODE_TYPE_LABELS[t]],
                     "mean_faithfulness": faith_means[NODE_TYPE_LABELS[t]],
                     "mean_attention": ""})
    for k, v in attn_means.items():
        rows.append({"group": "edge_type", "key": k,
                     "mean_saliency": "", "mean_faithfulness": "", "mean_attention": v})
    agg_write_csv(rows, os.path.join(exp_dir, "agg_explanations_summary.csv"),
                  fieldnames=["group", "key", "mean_saliency", "mean_faithfulness", "mean_attention"])

    agg_plot_bar(sal_means, "Mean Node Saliency by Node Type", "Saliency (Grad×Input)",
                 os.path.join(exp_dir, "agg_node_saliency.png"))
    agg_plot_bar(faith_means, "Mean Faithfulness by Node Type", "Δ Prob (deletion; proxy)",
                 os.path.join(exp_dir, "agg_node_faithfulness.png"))
    agg_plot_bar(attn_means, "Mean Edge Attention by Edge Type", "Attention (softmax)",
                 os.path.join(exp_dir, "agg_edge_attention.png"))

    agg_plot_heatmap(agg_edge_ntype, os.path.join(exp_dir, "agg_edge_ntype_heatmap.png"), normalize_rows=False)
    agg_plot_heatmap(agg_edge_ntype, os.path.join(exp_dir, "agg_edge_ntype_heatmap_row_norm.png"), normalize_rows=True)

    # Run summary
    with open(os.path.join(cfg["OUT_DIR"], "report_xai.json"), "w") as f:
        json.dump({
            "config": CONFIG,
            "val_metrics": val_metrics,
            "explained_indices": picked
        }, f, indent=2)

    print(f"[DONE] Artifacts saved under: {cfg['OUT_DIR']}")


if __name__ == "__main__":
    main()
