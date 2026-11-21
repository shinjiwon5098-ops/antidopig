FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SERVER_NAME=0.0.0.0 \
    PORT=8080 \
    GRADIO_ANALYTICS_ENABLED=False \
    NO_ALBUMENTATIONS_UPDATE=1 \
    MPLCONFIGDIR=/tmp/mpl \
    GRADIO_TEMP_DIR=/tmp/gradio

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch wheels to avoid multi-GB CUDA downloads in Cloud Run.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2.0" "torchvision>=0.17.0" && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /tmp/mpl /tmp/gradio && \
    python -c "import matplotlib.font_manager as fm; fm._load_fontmanager(try_read_cache=False)" || true

EXPOSE 8080

CMD ["python", "app.py"]
