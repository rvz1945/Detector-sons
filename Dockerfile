
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Instala o Deno, runtime JavaScript usado pelo yt-dlp
RUN curl -fsSL https://deno.land/install.sh | sh \
    && ln -s /root/.deno/bin/deno /usr/local/bin/deno

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -U -r requirements.txt

COPY . .

CMD sh -c 'uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}'
