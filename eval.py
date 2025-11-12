import os, argparse, yaml, torch, timm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from src.dataset import build_loaders

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--weights", type=str, default="runs/ckpt.pt")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f: cfg=yaml.safe_load(f)
    dl_tr, dl_va, n_classes = build_loaders(cfg["data_dir"], cfg["img_size"], cfg["batch_size"])
    device=get_device()
    model=timm.create_model(cfg["model"], pretrained=False, num_classes=n_classes).to(device)
    sd=torch.load(args.weights, map_location="cpu")
    if isinstance(sd,dict) and "state_dict" in sd: sd=sd["state_dict"]
    model.load_state_dict(sd, strict=False); model.eval()

    y_true=[]; y_pred=[]; y_prob=[]
    with torch.no_grad():
        for x,y in dl_va:
            x=x.to(device)
            probs=torch.softmax(model(x),dim=1).cpu().numpy()
            y_true+=y.numpy().tolist(); y_pred+=probs.argmax(1).tolist()
            if n_classes==2: y_prob+=probs[:,1].tolist()
    acc=accuracy_score(y_true,y_pred); print("Accuracy:", acc)
    print("Confusion:
", confusion_matrix(y_true,y_pred))
    print(classification_report(y_true,y_pred, digits=4))
    if n_classes==2 and len(y_prob)>0:
        try: print("AUROC:", roc_auc_score(y_true,y_prob))
        except Exception as e: print("AUROC failed:", e)

if __name__=="__main__":
    main()
