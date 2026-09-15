import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch_geometric.data import Data

# -------------------------------
# Config
# -------------------------------
IMG_DIR    = r"D:\Research\Topics\www\Datasets\The ISIC 2024 Challenge Dataset\ISIC_2024_Permissive_Training_Input"
META_FILE  = r"D:\Research\Topics\www\Datasets\The ISIC 2024 Challenge Dataset\ISIC p\metadata.csv"
SUPP_FILE  = r"D:\Research\Topics\www\Datasets\The ISIC 2024 Challenge Dataset\osdfs\ISIC_2024_Permissive_Training_Supplement.csv"
OUT_DIR    = r"D:\Research\Topics\www\Datasets\The ISIC 2024 Challenge Dataset\Output\ISIC P new"
BATCH_SIZE = 32
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

NODE_TYPE = {"organ":0, "tissue":1, "symptom":2, "patient":3}
EDGE_TYPE = {"part_of":0, "manifests_in":1, "co_dep":2, "adjacent_to":3, "has_finding":4}

# -------------------------------
# Step 1. Feature Extraction
# -------------------------------
def extract_features(img_paths):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]
    model = nn.Sequential(*modules).to(DEVICE)
    model.eval()

    feats, ids = [], []
    for i in tqdm(range(0,len(img_paths),BATCH_SIZE)):
        batch_paths = img_paths[i:i+BATCH_SIZE]
        imgs, good_ids = [], []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
                good_ids.append(os.path.splitext(os.path.basename(p))[0])
            except Exception as e:
                print(f"[WARN] Skipped {p}: {e}")
        if not imgs:
            continue
        imgs = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            f = model(imgs).squeeze(-1).squeeze(-1)  # [B,2048]
        feats.append(f.cpu().numpy())
        ids.extend(good_ids)
    return np.vstack(feats), ids

# -------------------------------
# Step 2. Metadata Processor
# -------------------------------
def process_metadata(meta_file, supp_file):
    df = pd.read_csv(meta_file)
    df = df.rename(columns={
        "isic_id": "image_id",
        "age_approx": "age",
        "anatom_site_general": "anatom_site",
        "clin_size_long_diam_mm": "size"
    })

    df_supp = pd.read_csv(supp_file, low_memory=False)
    df_supp["label"] = df_supp["iddx_full"].str.lower().map(
        lambda x: 1 if "melanoma" in str(x) or "malignant" in str(x) else 0
    )
    df_supp = df_supp[["isic_id","label"]].rename(columns={"isic_id":"image_id"})

    df = df.merge(df_supp, on="image_id", how="inner")
    df = df.dropna(subset=["image_id","patient_id","sex","anatom_site","label"])

    for col in ["age","size"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df["sex"] = df["sex"].str.lower().map(lambda x: 1 if str(x).startswith("m") else 0)

    site_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    site_encoded = site_encoder.fit_transform(df[["anatom_site"]])

    scaler = MinMaxScaler()
    df[["age","size"]] = scaler.fit_transform(df[["age","size"]])

    metadata_dict = {}
    for pos, row in enumerate(df.itertuples(index=False)):
        metadata_dict[row.image_id] = {
            "patient_id": row.patient_id,
            "age": float(row.age),
            "sex": float(row.sex),
            "site_onehot": site_encoded[pos],
            "size": float(row.size),
            "label": int(row.label)
        }
    return metadata_dict, site_encoder.categories_[0]

# -------------------------------
# Step 3. Graph Builder
# -------------------------------
def build_graph(img_id, cnn_feat, mrow):
    nodes = ["skin","visual_feat","age","sex","site","size","patient"]
    node_type = [
        NODE_TYPE["organ"],
        NODE_TYPE["symptom"],
        NODE_TYPE["symptom"],
        NODE_TYPE["symptom"],
        NODE_TYPE["symptom"],
        NODE_TYPE["symptom"],
        NODE_TYPE["patient"]
    ]

    feats = []
    feats.append(np.zeros(8,dtype=np.float32))
    feats.append(cnn_feat.astype(np.float32))
    feats.append(np.array([mrow["age"]],dtype=np.float32))
    feats.append(np.array([mrow["sex"]],dtype=np.float32))
    feats.append(mrow["site_onehot"].astype(np.float32))
    feats.append(np.array([mrow["size"]],dtype=np.float32))
    feats.append(np.zeros(8,dtype=np.float32))

    max_len = max(f.shape[0] for f in feats)
    feats = [np.pad(f,(0,max_len-f.shape[0])) for f in feats]
    x = torch.tensor(np.stack(feats,axis=0),dtype=torch.float32)

    E,T=[],[]
    E+=[[0,1],[1,0]]; T+=[EDGE_TYPE["manifests_in"],EDGE_TYPE["manifests_in"]]
    p=len(nodes)-1
    E+=[[p,1],[1,p]]; T+=[EDGE_TYPE["has_finding"],EDGE_TYPE["has_finding"]]
    for i in [2,3,4,5]:
        E+=[[p,i],[i,p]]
        T+=[EDGE_TYPE["has_finding"],EDGE_TYPE["has_finding"]]

    edge_index=torch.tensor(E,dtype=torch.long).t().contiguous()
    edge_type=torch.tensor(T,dtype=torch.long)
    y=torch.tensor([mrow["label"]],dtype=torch.long)

    return Data(
        x=x, edge_index=edge_index, edge_type=edge_type,
        node_type=torch.tensor(node_type,dtype=torch.long), y=y,
        node_labels=nodes
    )

# -------------------------------
# Step 4. Visualization
# -------------------------------
def visualize_graph_beautiful_static(data, save_path="isic_graph.png"):
    G=nx.Graph()
    labels={}
    for i,name in enumerate(data.node_labels):
        G.add_node(i,label=name,ntype=int(data.node_type[i]))
        labels[i]=name
    for (u,v),et in zip(data.edge_index.t().tolist(),data.edge_type.tolist()):
        G.add_edge(u,v,label=et)

    color_map=[]
    for n in G.nodes(data=True):
        ntype=n[1]['ntype']
        if ntype==NODE_TYPE["organ"]: color_map.append("#E74C3C")
        elif ntype==NODE_TYPE["symptom"]: color_map.append("#3498DB")
        elif ntype==NODE_TYPE["patient"]: color_map.append("#F1C40F")
        else: color_map.append("#95A5A6")

    pos=nx.spring_layout(G,seed=42)
    plt.figure(figsize=(10,8))
    nx.draw(G,pos,with_labels=True,labels=labels,
            node_color=color_map,node_size=1200,
            font_size=9,font_weight="bold",edge_color="gray")
    plt.title("ISIC 2024 Graph")
    plt.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close()
    print(f"[SAVED] PNG → {save_path}")

def visualize_graph_interactive_pyvis(data, html_path="isic_graph.html"):
    net=Network(height="800px",width="100%",bgcolor="#ffffff",font_color="#333")
    net.toggle_physics(True)
    for i,name in enumerate(data.node_labels):
        val=float(data.x[i].mean())
        color="#E74C3C" if int(data.node_type[i])==0 else "#3498DB" if int(data.node_type[i])==2 else "#F1C40F"
        net.add_node(i,label=name,title=f"{name}<br>Val:{val:.2f}",color=color,size=20)
    for (u,v),et in zip(data.edge_index.t().tolist(),data.edge_type.tolist()):
        net.add_edge(u,v,title=str(et),color="#7f8c8d")
    net.write_html(html_path)
    print(f"[SAVED] HTML → {html_path}")

# -------------------------------
# Step 5. Main Pipeline
# -------------------------------
def main():
    os.makedirs(OUT_DIR,exist_ok=True)

    # Extract features
    img_paths=[os.path.join(IMG_DIR,f) for f in os.listdir(IMG_DIR)
               if f.lower().endswith((".jpg",".png",".jpeg"))]
    print(f"[INFO] Found {len(img_paths)} images")
    feats,ids=extract_features(img_paths)

    # Process metadata + labels
    metadata_dict,_=process_metadata(META_FILE,SUPP_FILE)

    # Build graphs
    graphs=[]
    for i,(img_id,feat) in enumerate(zip(ids,feats)):
        if img_id not in metadata_dict:
            continue
        mrow=metadata_dict[img_id]
        g=build_graph(img_id,feat,mrow)
        graphs.append(g)

        if i==0:  # only visualize first graph
            visualize_graph_beautiful_static(g,save_path=os.path.join(OUT_DIR,"isic_00000.png"))
            visualize_graph_interactive_pyvis(g,html_path=os.path.join(OUT_DIR,"isic_00000.html"))

    # Save all graphs in a single file
    out_path = os.path.join(OUT_DIR,"isic_dataset.pt")
    torch.save(graphs, out_path)

    print(f"[DONE] Built {len(graphs)} ISIC 2024 graphs → {out_path}")

if __name__=="__main__":
    main()
