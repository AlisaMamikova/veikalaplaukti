import os, argparse, yaml, torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm.auto import tqdm
import timm
from src.dataset import build_loaders
from src.utils import save_ckpt

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def get_device():
    # Ja ir pieejams CUDA, tad lietojiet CUDA
    # Ja ir pieejams MPS (Metal Performance Shaders), tad lietojiet MPS
    # Ja nevar izmantot nākamo, tad lietojiet CPU
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    args = ap.parse_args()

    with open(args.config) as f: cfg = yaml.safe_load(f)
    for k,v in vars(args).items():
        if v is not None and k!="config":
            key = {"img_size":"img_size","batch":"batch_size","data_dir":"data_dir"}.get(k,k)
            cfg[key] = v

    dl_tr, dl_va, n_classes = build_loaders(cfg["data_dir"], cfg["img_size"], cfg["batch_size"], cfg.get("num_workers",4))
    device = get_device(); print("Using device:", device)

    model = timm.create_model(cfg["model"], pretrained=True, num_classes=n_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    os.makedirs(cfg["out_dir"], exist_ok=True)

    best = -1e9
    for ep in range(cfg["epochs"]):
        model.train(); run_loss=0.0
        pbar = tqdm(dl_tr, total=len(dl_tr), desc=f"Epoch {ep+1}/{cfg['epochs']} [train]", unit="batch")
        for i,(x,y) in enumerate(pbar,1):
            x=x.to(device, non_blocking=(device.type=='cuda')); y=y.to(device, non_blocking=(device.type=='cuda'))
            logits=model(x); loss=crit(logits,y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run_loss += loss.item(); pbar.set_postfix(loss=f"{run_loss/i:.4f}")

        model.eval(); y_true=[]; y_pred=[]; y_prob=[]
        with torch.no_grad():
            pbar_v = tqdm(dl_va, total=len(dl_va), desc=f"Epoch {ep+1}/{cfg['epochs']} [val]", unit="batch")
            for x,y in pbar_v:
                x=x.to(device, non_blocking=(device.type=='cuda'))
                probs=torch.softmax(model(x), dim=1).cpu().numpy()
                y_true.extend(y.numpy().tolist()); y_pred.extend(probs.argmax(1).tolist())
                if n_classes==2: y_prob.extend(probs[:,1].tolist())
        acc=accuracy_score(y_true,y_pred)
        if n_classes==2:
            try: auc=roc_auc_score(y_true,y_prob)
            except Exception: auc=float("nan")
            score = auc if auc==auc else acc
            print(f"Epoch {ep+1}/{cfg['epochs']}  ACC={acc:.4f}  AUC={auc:.4f}")
        else:
            score=acc; print(f"Epoch {ep+1}/{cfg['epochs']}  ACC={acc:.4f}")

        if score>best:
            best=score; save_ckpt(model, f"{cfg['out_dir']}/ckpt.pt")

if __name__ == "__main__":
    main()
