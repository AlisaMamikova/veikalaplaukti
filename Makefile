.PHONY: setup train eval app docker-build docker-run

setup:
	python -m pip install -r requirements.txt

train:
	python train.py --config configs/config.yaml

eval:
	python eval.py --config configs/config.yaml --weights runs/ckpt.pt

app:
	streamlit run app.py --server.port=8501 --server.address=0.0.0.0

docker-build:
	docker build -t $(shell basename $(PWD)):latest .

docker-run:
	docker run --rm -it -p 8501:8501 -v $(PWD)/data:/app/data -v $(PWD)/runs:/app/runs $(shell basename $(PWD)):latest
