# PKLot Parking Occupancy — Classification (timm + PyTorch)

Train a classifier to predict **parking spot occupancy** (empty / occupied) from cropped images.
Comes with a **Streamlit demo** (crops or full image + ROI rectangles) and **Docker**.

## Quickstart
```bash
pip install -r requirements.txt
python train.py --config configs/config.yaml
streamlit run app.py
```
