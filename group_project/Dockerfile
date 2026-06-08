FROM python:3.11-slim

WORKDIR /app

# Cài system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước (tận dụng Docker cache)
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . .

# HF Spaces yêu cầu port 7860
EXPOSE 7860

# Chạy Streamlit
CMD ["streamlit", "run", "chatbot.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
