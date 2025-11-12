import json

coco_path = "data/valid/_annotations.coco.json"   # vai train/test
image_name = "2012-12-15_09_35_05_jpg.rf.76e8d861cbbcbf5a4ecf71ddbef18dc3.jpg"                     # precīzs faila nosaukums

with open(coco_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

# atrod attēla ID pēc file_name
img = next(im for im in coco["images"] if im["file_name"] == image_name)
img_id = img["id"]

# savāc visus bbox šai bildei
rois = []
for i, ann in enumerate(coco["annotations"], start=1):
    if ann.get("image_id") == img_id and "bbox" in ann:
        x, y, w, h = ann["bbox"]
        rois.append({"id": i, "rect": [int(x), int(y), int(w), int(h)]})

out = {"rois": rois}
with open("data/parking_rois.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("Saved parking_rois.json with", len(rois), "ROIs")