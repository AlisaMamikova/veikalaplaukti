FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \    build-essential \    libgl1 \    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Install CPU wheels for PyTorch (smaller, reliable)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \    torch torchvision torchaudio && \    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
