FROM python:3.10-slim

WORKDIR /code

# Install system audio dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY backend/ /code/

ENV PORT=7860
EXPOSE 7860

CMD ["python", "main.py"]
