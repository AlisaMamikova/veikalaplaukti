
#python scripts/subsample_coco.py --data data --out data_small --train 100 --valid 20 --test 50

import argparse, json, os, shutil, random
from pathlib import Path
from typing import Dict, List, Tuple, Set

def find_val_dir(data_dir: Path) -> Path:
    if (data_dir / "valid").exists():
        return data_dir / "valid"
    if (data_dir / "val").exists():
        return data_dir / "val"
    raise FileNotFoundError(f"Neither '{data_dir/'valid'}' nor '{data_dir/'val'}' exists.")

def load_coco(split_dir: Path) -> Dict:
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {ann_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_coco(coco: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def copy_or_link(src: Path, dst: Path, use_copy: bool):
    ensure_parent(dst)
    try:
        if use_copy:
            shutil.copy2(src, dst)
        else:
            # hardlink for space saving; fall back to copy if FS does not support
            os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def sample_images(images: List[Dict], k: int, seed: int) -> List[Dict]:
    """Uniform random sampling of image entries (by image objects)."""
    if k >= len(images):
        return images[:]  # keep all
    rng = random.Random(seed)
    idxs = list(range(len(images)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:k])  # stable order for reproducibility
    return [images[i] for i in idxs]

def filter_coco_by_images(coco: Dict, keep_image_ids: Set[int]) -> Dict:
    out = {}
    # Preserve meta fields if present
    for k in ["info", "licenses", "categories"]:
        if k in coco:
            out[k] = coco[k]
    # Filter images and annotations
    img_id_set = set(int(i) for i in keep_image_ids)
    out["images"] = [im for im in coco.get("images", []) if int(im.get("id")) in img_id_set]
    out["annotations"] = [a for a in coco.get("annotations", []) if int(a.get("image_id")) in img_id_set]
    return out

def subsample_split(split_name: str, src_dir: Path, dst_dir: Path, k_images: int, seed: int, copy_files: bool, inplace: bool):
    if not src_dir.exists():
        print(f"[{split_name}] skip (not found): {src_dir}")
        return

    coco = load_coco(src_dir)
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    sel_images = sample_images(images, k_images, seed)
    keep_ids = {int(im["id"]) for im in sel_images}
    coco_new = filter_coco_by_images(coco, keep_ids)

    # Destination directories
    if inplace:
        # Remove files not selected and overwrite annotations in-place
        kept_rel_paths = set()
        for im in coco_new["images"]:
            rel = im["file_name"]
            kept_rel_paths.add(rel)

        # Delete unselected image files
        removed = 0
        for im in images:
            rel = im["file_name"]
            if rel not in kept_rel_paths:
                p = src_dir / rel
                if p.exists() and p.is_file():
                    try:
                        p.unlink()
                        removed += 1
                    except Exception as e:
                        print(f"  warn: could not remove {p}: {e}")
        # Write new annotations
        write_coco(coco_new, src_dir / "_annotations.coco.json")
        print(f"[{split_name}] kept images: {len(sel_images)}/{len(images)}, "
              f"kept anns: {len(coco_new['annotations'])}/{len(anns)}, removed files: {removed}")
    else:
        # Copy/link selected images into dst_dir and write new annotations there
        for im in sel_images:
            rel = Path(im["file_name"])
            src_img = src_dir / rel
            dst_img = dst_dir / rel
            copy_or_link(src_img, dst_img, copy_files)
        write_coco(coco_new, dst_dir / "_annotations.coco.json")
        print(f"[{split_name}] kept images: {len(sel_images)}/{len(images)}, "
              f"kept anns: {len(coco_new['annotations'])}/{len(anns)} "
              f"-> {dst_dir}")

def main():
    ap = argparse.ArgumentParser(description="Subsample COCO dataset per split (train/valid(or val)/test).")
    ap.add_argument("--data", type=str, default="data", help="Root with train/, valid/ or val/, test/")
    ap.add_argument("--out", type=str, default="data_small", help="Output root for subsampled dataset (ignored with --inplace)")
    ap.add_argument("--train", type=int, default=100, help="Number of train images to keep")
    ap.add_argument("--valid", type=int, default=20, help="Number of valid/val images to keep")
    ap.add_argument("--test", type=int, default=50, help="Number of test images to keep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of hardlinking")
    ap.add_argument("--inplace", action="store_true", help="Modify in place (delete unselected files and rewrite annotations)")
    args = ap.parse_args()

    data_dir = Path(args.data).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data root not found: {data_dir}")

    train_dir = data_dir / "train"
    val_dir = find_val_dir(data_dir)
    test_dir = data_dir / "test"

    if args.inplace:
        subsample_split("train", train_dir, None, args.train, args.seed, args.copy, inplace=True)
        if val_dir.exists():
            subsample_split(val_dir.name, val_dir, None, args.valid, args.seed, args.copy, inplace=True)
        if test_dir.exists():
            subsample_split("test", test_dir, None, args.test, args.seed, args.copy, inplace=True)
    else:
        out_dir = Path(args.out).resolve()
        (out_dir / "train").mkdir(parents=True, exist_ok=True)
        if (data_dir / "valid").exists():
            (out_dir / "valid").mkdir(parents=True, exist_ok=True)
        elif (data_dir / "val").exists():
            (out_dir / "val").mkdir(parents=True, exist_ok=True)
        if test_dir.exists():
            (out_dir / "test").mkdir(parents=True, exist_ok=True)

        subsample_split("train", train_dir, out_dir / "train", args.train, args.seed, args.copy, inplace=False)
        if val_dir.exists():
            subsample_split(val_dir.name, val_dir, out_dir / val_dir.name, args.valid, args.seed, args.copy, inplace=False)
        if test_dir.exists():
            subsample_split("test", test_dir, out_dir / "test", args.test, args.seed, args.copy, inplace=False)

if __name__ == "__main__":
    main()