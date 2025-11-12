# app.py
import os, json
import streamlit as st
import numpy as np
from PIL import Image
import torch, timm
from torchvision import transforms
import cv2
import pandas as pd  # tikai Streamlit tabulām; var izņemt, ja nevajag

st.set_page_config(page_title="PKLot Parking Occupancy", layout="wide")
st.title("Parking Occupancy — COCO categories (timm)")

# ---- Controls ----
weights = st.text_input("Weights (.pt state_dict)", "runs/ckpt.pt")
model_name = st.text_input("timm model", "tf_efficientnet_b0.ns_jft_in1k")
img_size = st.number_input("Image size", 64, 1024, 224, 32)
device_choice = st.selectbox("Device", ["auto","cuda","mps","cpu"], index=0)

coco_path = st.text_input("COCO annotations path (to get class names)",
                          "data/valid/_annotations.coco.json")

mode = st.radio("Mode", ["Cropped spot image(s)", "Full lot image + ROI JSON"])

if mode == "Cropped spot image(s)":
    uploads = st.file_uploader("Upload crop(s)", type=["jpg","jpeg","png"], accept_multiple_files=True)
else:
    lot_img = st.file_uploader("Upload parking-lot image", type=["jpg","jpeg","png"])
    roi_json = st.file_uploader("Upload ROI JSON (rects)", type=["json"])

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def pick_device(d):
    if d != "auto":
        return d
    try:
        if torch.cuda.is_available(): return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    except Exception:
        pass
    return "cpu"

def infer_classes_from_coco(path: str):
    """
    Atgriež klasēs no COCO 'categories'. Ja ir vecākklase 'spaces', ņem tās bērnus
    (piem., 'space-empty', 'space-occupied'). Pretējā gadījumā — visas kategorijas,
    izņemot tieši 'spaces'.
    """
    if not path or not os.path.exists(path):
        return ["space-empty", "space-occupied"]  # drošais noklusējums
    try:
        with open(path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        cats = coco.get("categories", [])
        cats = sorted(cats, key=lambda c: int(c.get("id", 0)))  # stabila kārtība pēc id
        child = [c["name"] for c in cats if str(c.get("supercategory","")).lower() == "spaces" and c.get("name")]
        if child:
            return child
        return [c["name"] for c in cats if str(c.get("name","")).lower() != "spaces"]
    except Exception as e:
        st.warning(f"Could not read COCO categories: {e}")
        return ["space-empty", "space-occupied"]

def load_model(weights, model_name, class_names, image_size, device):
    device = pick_device(device)
    st.info(f"Using device: {device}")
    nc = int(len(class_names))

    model = timm.create_model(model_name, pretrained=False, num_classes=nc)
    # robusta state_dict ielāde: atmet neatbilstošos svarus (piem., classifier no cita nc)
    try:
        sd = torch.load(weights, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        msd = model.state_dict()
        filt = {k: v for k, v in sd.items() if k in msd and tuple(v.shape) == tuple(msd[k].shape)}
        model.load_state_dict(filt, strict=False)
        st.success(f"Loaded weights (kept {len(filt)}/{len(msd)} keys)")
    except Exception as e:
        st.warning(f"Could not load weights: {e}. Using random init.")

    model.eval().to(device)
    tfm = transforms.Compose([
        transforms.Resize((int(image_size), int(image_size))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return model, tfm, device

def predict(model, tfm, device, pil_img):
    x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    return probs

def color_for(label: str):
    # zaļš -> empty, sarkans -> occupied
    low = label.lower()
    if "empty" in low: return (0, 220, 0)
    if "occupied" in low: return (0, 0, 220)
    return (255, 165, 0)  # oranžs, ja kas cits

# ---- Classes from COCO ----
class_names = infer_classes_from_coco(coco_path)
st.write("Classes:", ", ".join(class_names))

# ---- UI modes ----
if mode == "Cropped spot image(s)":
    if uploads:
        model, tfm, device = load_model(weights, model_name, class_names, int(img_size), device_choice)
        cols = st.columns(min(3, len(uploads)))
        for i, up in enumerate(uploads):
            img = Image.open(up).convert("RGB")
            probs = predict(model, tfm, device, img)
            pred = int(np.argmax(probs))
            with cols[i % len(cols)]:
                st.image(img, caption=f"{class_names[pred]} ({probs[pred]*100:.1f}%)", use_container_width=True)
                st.dataframe(pd.DataFrame({"class": class_names, "prob": probs}))
else:
    st.markdown("""**ROI JSON format (rectangles):**
```json
{ "rois": [ {"id": 1, "rect": [x,y,w,h]}, {"id": 2, "rect": [x,y,w,h]} ] }
```""")
    if lot_img and roi_json:
        img = Image.open(lot_img).convert("RGB")
        model, tfm, device = load_model(weights, model_name, class_names, int(img_size), device_choice)
        try:
            rois = json.load(roi_json).get("rois", [])
        except Exception as e:
            st.error(f"Invalid ROI JSON: {e}")
            rois = []

        vis = np.array(img)[:, :, ::-1].copy()  # BGR for cv2
        counts = {cn: 0 for cn in class_names}

        for roi in rois:
            rect = roi.get("rect")
            if not rect or len(rect) != 4: continue
            x, y, w, h = map(int, rect)
            crop = img.crop((x, y, x + w, y + h))
            probs = predict(model, tfm, device, crop)
            pred = int(np.argmax(probs))
            label = class_names[pred]
            counts[label] = counts.get(label, 0) + 1

            color = color_for(label)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, f"{label} {probs[pred]*100:.0f}%", (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        st.subheader("Overlay")
        st.image(vis[:, :, ::-1], use_container_width=True, caption="Predicted occupancy")

        st.subheader("Counts")
        st.json(counts)