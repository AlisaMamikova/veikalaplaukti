# src/dataset.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def _clamp_bbox(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    """Clamp COCO bbox [x,y,w,h] to image bounds and return integer (x1,y1,x2,y2)."""
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(W, int(round(x + w)))
    y2 = min(H, int(round(y + h)))
    # ensure valid box
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


class COCOROIDataset(Dataset):
    """
    COCO-style ROI → klasifikācijas crops no bbox.
    Sagaida split_dir ar attēliem un '_annotations.coco.json'.
    """
    def __init__(
        self,
        split_dir: Path,
        img_size: int = 224,
        is_train: bool = True,
        classes: Optional[List[str]] = None,
        min_area: int = 16,
    ):
        self.split_dir = Path(split_dir)
        self.ann_path = self.split_dir / "_annotations.coco.json"
        if not self.ann_path.exists():
            raise FileNotFoundError(f"COCO annotations file not found: {self.ann_path}")

        with open(self.ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # images: id -> file_name
        img_id_to_fname: Dict[int, str] = {img["id"]: img["file_name"] for img in coco.get("images", [])}
        # categories: id -> name
        cat_id_to_name: Dict[int, str] = {c["id"]: c["name"] for c in coco.get("categories", [])}

        # Fix/derive classes mapping (consistent train/val)
        if classes is None:
            # Unique category names sorted for stable mapping
            names = sorted(set(cat_id_to_name.values()))
            self.classes = names
        else:
            self.classes = list(classes)

        name_to_idx = {n: i for i, n in enumerate(self.classes)}

        # Build item list: (image_path, (x1,y1,x2,y2), class_idx)
        items: List[Tuple[Path, Tuple[int, int, int, int], int]] = []
        for ann in coco.get("annotations", []):
            cid = ann.get("category_id")
            cname = cat_id_to_name.get(cid, None)
            if cname is None or cname not in name_to_idx:
                # unknown category → skip
                continue
            img_id = ann.get("image_id")
            fname = img_id_to_fname.get(img_id, None)
            if fname is None:
                continue
            img_path = self.split_dir / fname
            if not img_path.exists():
                # skip missing image
                continue
            bbox = ann.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            if w * h < float(min_area):
                continue

            # Clamp bbox to image
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                # if can't open here, still push path; open later lazily
                # but we need W,H to clamp; fallback to naive int cast
                W = H = 10**9
            x1, y1, x2, y2 = _clamp_bbox(x, y, w, h, W, H)
            items.append((img_path, (x1, y1, x2, y2), name_to_idx[cname]))

        if len(items) == 0:
            raise ValueError(f"No items found after parsing {self.ann_path}. "
                             f"Check that annotations contain bboxes and categories present in dataset classes.")

        self.items = items

        # Transforms (ImageNet norm for timm)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if is_train:
            self.tfms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.tfms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        img_path, (x1, y1, x2, y2), y = self.items[i]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            # robust crop (in case bbox slightly out of bounds)
            W, H = im.size
            x1_c = max(0, min(x1, W - 1))
            y1_c = max(0, min(y1, H - 1))
            x2_c = max(1, min(x2, W))
            y2_c = max(1, min(y2, H))
            crop = im.crop((x1_c, y1_c, x2_c, y2_c))
        x = self.tfms(crop)
        return x, torch.tensor(y, dtype=torch.long)


def _pick_val_dir(data_dir: Path) -> Path:
    """Support both 'valid' and 'val'."""
    if (data_dir / "valid").exists():
        return data_dir / "valid"
    if (data_dir / "val").exists():
        return data_dir / "val"
    raise FileNotFoundError(f"Neither '{data_dir / 'valid'}' nor '{data_dir / 'val'}' exists.")


def build_loaders(
    data_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    min_area: int = 16,
):
    """
    Izveido DataLoaderus no COCO anotācijām katrā split mapē.
    - data_dir/train/_annotations.coco.json
    - data_dir/valid/_annotations.coco.json  (vai data_dir/val/…)
    - (pēc izvēles) data_dir/test/_annotations.coco.json  — netiek atgriezts
    """
    data_dir = Path(data_dir)

    train_dir = data_dir / "train"
    val_dir = _pick_val_dir(data_dir)

    # Uzbūvē train, tad val ar tādu pašu klašu kartējumu
    ds_tr_tmp = COCOROIDataset(train_dir, img_size=img_size, is_train=True, classes=None, min_area=min_area)
    classes = ds_tr_tmp.classes  # noteikts no train kategoriju nosaukumiem

    ds_tr = ds_tr_tmp  # jau ir
    ds_va = COCOROIDataset(val_dir, img_size=img_size, is_train=False, classes=classes, min_area=min_area)

    train_dl = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=False)
    val_dl = DataLoader(ds_va, batch_size=batch_size * 2, shuffle=False,
                        num_workers=num_workers, pin_memory=False)

    n_classes = len(classes)
    return train_dl, val_dl, n_classes