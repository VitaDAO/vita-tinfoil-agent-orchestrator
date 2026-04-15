FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py ./
COPY qmd-cert.pem ./

RUN mkdir -p /home/user && useradd -m -d /home/user user && chown -R user:user /app /home/user

ENV HOME=/home/user

USER user
EXPOSE 8080

CMD ["python", "/app/bot.py"]
